"""

Pydantic models for Automatic Speech Recognition (ASR) and Text-To-Speech (TTS) responses.
Response types for TTS audio.

This module provides response types for audio with full OpenAI compatibility:
  • TTSResponse: TTS response containing audio bytes and text encoding.
  • ASRResponse: ASR response containing transcription results.
  • ChunkingStrategy: Server VAD configuration for audio chunking.

"""

import json
import os
from typing import (
    Annotated,
    Any,
    AsyncContextManager,
    AsyncIterator,
    Callable,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Required,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
)

import aiofiles

from pydantic import TypeAdapter, ValidationError

from air.types.base import CustomBaseModel


class TTSResponse:
    """
    Response wrapper for TTS audio data.

    Mimics OpenAI's HttpxBinaryResponseContent for full compatibility
    with OpenAI's client interface. Provides both sync and async methods
    for reading and streaming audio content.

    Attributes:
        content: Raw audio bytes from TTS synthesis.
        encoding: Text encoding used for the content.
    """

    def __init__(self, content: bytes, encoding: str = "utf-8"):
        """
        Initialize TTSResponse with audio data.

        Args:
            content: Raw audio bytes from TTS synthesis
            encoding: Text encoding (default: utf-8)
        """
        self._content = content
        self._encoding = encoding

    @property
    def content(self) -> bytes:
        """Raw audio bytes."""
        return self._content

    @property
    def text(self) -> str:
        """Text representation of the content."""
        return self._content.decode(self._encoding)

    @property
    def encoding(self) -> str:
        """Content encoding."""
        return self._encoding

    @property
    def charset_encoding(self) -> str:
        """Charset encoding."""
        return self._encoding

    def read(self) -> bytes:
        """Read the complete audio data."""
        return self._content

    def __enter__(self):
        """Enter sync context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit sync context manager."""
        pass

    def json(self, **kwargs: Any) -> Any:
        """Parse content as JSON (not applicable for audio, but maintaining interface)."""

        return json.loads(self.text, **kwargs)

    def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
        """Iterate over content in byte chunks.

        Args:
            chunk_size: Size of each chunk in bytes (default: 1024).

        Yields:
            bytes: Audio data chunks.
        """
        if chunk_size is None:
            chunk_size = 1024

        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def iter_text(self, chunk_size: int | None = None) -> Iterator[str]:
        """Iterate over content as text chunks.

        Note: Not applicable for audio data, but maintained for interface compatibility.

        Args:
            chunk_size: Size of each chunk in bytes (default: 1024).

        Yields:
            str: Decoded text chunks.
        """
        for chunk in self.iter_bytes(chunk_size):
            yield chunk.decode(self._encoding)

    def iter_lines(self) -> Iterator[str]:
        """Iterate over content line by line.

        Note: Not applicable for audio data, but maintained for interface compatibility.

        Yields:
            str: Lines of decoded text.
        """
        return iter(self.text.splitlines())

    def iter_raw(self, chunk_size: int | None = None) -> Iterator[bytes]:
        """Iterate over raw content in chunks.

        Args:
            chunk_size: Size of each chunk in bytes (default: 1024).

        Yields:
            bytes: Raw audio data chunks.
        """
        return self.iter_bytes(chunk_size)

    def write_to_file(self, file: str | os.PathLike[str]) -> None:
        """
        Write the audio output to the given file.

        Args:
            file: Filename or path-like object where to save the audio
        """
        with open(file, mode="wb") as f:
            f.write(self._content)

    def stream_to_file(
        self,
        file: str | os.PathLike[str],
        *,
        chunk_size: int | None = None,
    ) -> None:
        """
        Stream content to file in chunks.

        Args:
            file: Filename or path-like object
            chunk_size: Size of chunks to write
        """
        with open(file, mode="wb") as f:
            for chunk in self.iter_bytes(chunk_size):
                f.write(chunk)

    def close(self) -> None:
        """Close the response."""
        pass

    # Async methods to match OpenAI's interface
    async def aread(self) -> bytes:
        """Async read of the complete audio data."""
        return self._content

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        pass

    async def aiter_bytes(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        """Async iteration over content in byte chunks.

        Args:
            chunk_size: Size of each chunk in bytes (default: 1024).

        Yields:
            bytes: Audio data chunks.
        """
        if chunk_size is None:
            chunk_size = 1024

        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    async def aiter_text(self, chunk_size: int | None = None) -> AsyncIterator[str]:
        """Async iteration over content as text chunks.

        Args:
            chunk_size: Size of each chunk in bytes (default: 1024).

        Yields:
            str: Decoded text chunks.
        """
        async for chunk in self.aiter_bytes(chunk_size):
            yield chunk.decode(self._encoding)

    async def aiter_lines(self) -> AsyncIterator[str]:
        """Async iteration over content line by line.

        Yields:
            str: Lines of decoded text.
        """
        for line in self.text.splitlines():
            yield line

    async def aiter_raw(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        """Async iteration over raw content in chunks.

        Args:
            chunk_size: Size of each chunk in bytes (default: 1024).

        Yields:
            bytes: Raw audio data chunks.
        """
        async for chunk in self.aiter_bytes(chunk_size):
            yield chunk

    async def astream_to_file(
        self,
        file: str | os.PathLike[str],
        *,
        chunk_size: int | None = None,
    ) -> None:
        """
        Async stream content to file in chunks.

        Args:
            file: Filename or path-like object
            chunk_size: Size of chunks to write
        """
        async with aiofiles.open(file, mode="wb") as f:
            async for chunk in self.aiter_bytes(chunk_size):
                await f.write(chunk)

    async def aclose(self) -> None:
        """Async close the response"""
        pass


class ASRResponse(CustomBaseModel):
    """Top-level Automatic Speech Recognition response returned by the API.

    Attributes:
        text (Union[str, None]): The transcription of the audio file
        success (bool): Whether the transcription request was successful
        error (Optional[str]): Optional error message if transcription was not successful
        confidence (Optional[float]): Optional confidence of the tokens
    """

    text: Union[str, None]
    success: bool
    error: Optional[str] = None
    confidence: Optional[float] = None


class ChunkingStrategy(TypedDict, total=False):
    """
    Controls how the audio is cut into chunks.

    Attributes:
        type (Literal["server_vad"]): Selects server-side VAD chunking (required).
        prefix_padding_ms (int, optional):  Lead-in context before speech, 0–5000 ms.
        silence_duration_ms(int, optional): Trailing silence that closes a chunk, 0–5000 ms.
        threshold (float, optional): VAD sensitivity, 0.0–1.0 (currently ignored).
    """

    type: Required[Literal["server_vad"]]
    prefix_padding_ms: int  # initial_vad_delay, 0 - 5000 ms
    silence_duration_ms: int  # segmentation_vad_delay, 0 - 5000 ms
    threshold: float  # vad_sensitivity, 0.0 - 1.0 : NotImplemented


class Logprob(CustomBaseModel):
    """
    Represents the log probability data for a specific token in a transcription
    or language model output.

    Attributes:
        token (Optional[str]): The text token for which the log probability was computed.
        bytes (Optional[List[int]]): The raw byte representation of the token,
            useful for non-text or multilingual tokens.
        logprob (Optional[float]): The log probability value associated with the token,
            indicating model confidence.
    """

    token: Optional[str] = None
    bytes: Optional[List[int]] = None
    logprob: Optional[float] = None


class TranscriptionTextDeltaEvent(CustomBaseModel):
    """
    Represents an incremental transcription update event emitted during streaming transcription.

    This event provides a new segment of transcribed text (delta) as it becomes available,
    allowing for real-time transcription updates. It may optionally include token-level
    log probabilities if requested.

    Attributes:
        delta (str): The text delta that was additionally transcribed.
        type (Literal["transcript.text.delta"]): Type of the event. Always "transcript.text.delta".
        logprobs (Optional[List[Logprob]]): The log probabilities of the tokens in the delta.
    """

    delta: str
    type: Literal["transcript.text.delta"]
    logprobs: Optional[List[Logprob]] = None


class TranscriptionTextDoneEvent(CustomBaseModel):
    """
    Represents the final transcription result emitted at the end of the audio input.

    This event marks the completion of the transcription stream and contains the full
    transcribed text. It may optionally include token-leveln log probabilities if requested.

    Attributes:
        text (str): The text that was transcribed.
        type (Literal["transcript.text.done"]): Type of the event. Always "transcript.text.done".
        logprobs (Optional[List[Logprob]]): The log probability of each token in the transcription.
    """

    text: str
    type: Literal["transcript.text.done"]
    logprobs: Optional[List[Logprob]] = None


# A union type representing streaming transcription delta/done events, used for parsing.
# The `type` field is used as a discriminator to determine the event variant.
TranscriptionStreamEvent: TypeAlias = Annotated[
    Union[TranscriptionTextDeltaEvent, TranscriptionTextDoneEvent],
    {"discriminator": "type"},
]


T_co = TypeVar("T_co", covariant=True)


class Stream(Protocol[T_co]):  # pylint: disable=too-few-public-methods
    """
    Protocol for synchronous streaming responses.

    Represents any object that supports iteration over items of type `T`,
    typically used for non-async streaming data.
    """

    def __iter__(self) -> Iterator[T_co]: ...


class AsyncStream(Protocol[T_co]):  # pylint: disable=too-few-public-methods
    """
    Protocol for asynchronous streaming responses.

    Represents any object that supports async iteration over items of type `T`,
    typically used for consuming async streaming data.
    """

    def __aiter__(self) -> AsyncIterator[T_co]: ...


class APIResponse(Generic[T_co]):
    """
    Represents a synchronous API response with optional streaming support.

    Wraps the parsed result and underlying HTTP response object. Provides
    utility methods to iterate over raw or parsed response lines.
    """

    def __init__(self, parsed: T_co, http_response: Any):
        self.parsed = parsed
        self.http_response = http_response

    def iter_lines(self) -> Iterator[str]:
        """
        Yield raw `data:` lines if the server streamed SSE,
        otherwise yield exactly one JSON line representing the
        non-stream response.
        """
        # 1) streaming → relay the socket line-by-line
        if str(self.http_response.headers.get("content-type", "")).startswith(
            "text/event-stream"
        ):
            for raw in self.http_response.iter_lines():
                yield raw.decode().rstrip("\r\n")
            return

        # 2) non-stream → fake a single line so caller code still works
        if isinstance(self.parsed, CustomBaseModel):
            payload = self.parsed.model_dump_json()
        else:  # plain dataclass / dict
            payload = json.dumps(self.parsed)
        yield payload

    def parse(self) -> Iterator[Union["TranscriptionStreamEvent", "ASRResponse"]]:
        """
        Yield `TranscriptionStreamEvent` objects if the server streamed SSE,
        otherwise yield exactly one parsed event representing the
        non-stream response.
        """
        # 1) streaming → relay the socket line-by-line
        if str(self.http_response.headers.get("content-type", "")).startswith(
            "text/event-stream"
        ):
            for raw in self.http_response.iter_lines():
                line = raw.decode().rstrip("\r\n")
                if not line.startswith("data:"):
                    continue  # skip comments
                if line.strip() == "data: [DONE]":
                    break
                try:
                    event = TypeAdapter(TranscriptionStreamEvent).validate_python(
                        json.loads(line[5:].strip())
                    )
                    yield event
                except Exception as e:
                    print("invalid payload:", e)
                    continue
            return

        # 2) non-stream → fake a single event so caller code still works
        if isinstance(self.parsed, CustomBaseModel):
            payload = self.parsed.model_dump_json()
        else:  # plain dataclass / dict
            payload = json.dumps(self.parsed)
        yield ASRResponse.model_validate(json.loads(payload))

    def __repr__(self):
        return f"<APIResponse parsed={self.parsed!r}, http_response={self.http_response!r}>"


class AsyncAPIResponse(Generic[T_co]):
    """
    Represents an asynchronous API response with optional streaming support.

    Wraps the parsed result and underlying async HTTP response object.
    Provides async utilities to iterate over response content.
    """

    def __init__(self, parsed: T_co, http_response: Any):
        self.parsed = parsed
        self.http_response = http_response

    async def iter_lines(self) -> AsyncIterator[str]:
        """
        Yield raw `data:` lines if the server streamed SSE,
        otherwise yield exactly one JSON line representing the
        non-stream response.
        """
        # 1) streaming → relay the socket line-by-line
        if self.http_response.headers.get("content-type", "").startswith(
            "text/event-stream"
        ):
            async for raw in self.http_response.content:
                yield raw.decode().rstrip("\r\n")
            return

        # 2) non-stream → fake a single line so caller code still works
        if isinstance(self.parsed, CustomBaseModel):
            payload = self.parsed.model_dump_json()
        else:  # plain dataclass / dict
            payload = json.dumps(self.parsed)
        yield payload

    async def parse(
        self,
    ) -> AsyncIterator["Union[TranscriptionStreamEvent, ASRResponse]"]:
        """
        Yield raw `data:` lines if the server streamed SSE,
        otherwise yield exactly one JSON line representing the
        non-stream response.
        """
        # 1) streaming → relay the socket line-by-line
        if self.http_response.headers.get("content-type", "").startswith(
            "text/event-stream"
        ):
            async for raw in self.http_response.content:
                raw = raw.decode().rstrip("\r\n")
                if not raw.startswith("data:"):
                    continue  # skip comments / blanks
                if raw.startswith("data: [DONE]"):
                    break
                try:
                    event = TypeAdapter(TranscriptionStreamEvent).validate_python(
                        json.loads(raw[5:].strip())
                    )
                    yield event
                except Exception as e:
                    print("invalid payload:", e)
                    continue
            return

        # 2) non-stream → fake a single line so caller code still works
        if isinstance(self.parsed, CustomBaseModel):
            payload = self.parsed.model_dump_json()
        else:  # plain dataclass / dict
            payload = json.dumps(self.parsed)
        yield ASRResponse.model_validate(json.loads(payload))

    def __repr__(self):
        return f"<APIResponse parsed={self.parsed!r}, http_response={self.http_response!r}>"


class ResponseContextManager(Generic[T_co]):
    """
    A synchronous context manager wrapper for API responses.

    Executes a user-provided `enter()` function when the context is entered.
    Used to encapsulate setup and teardown behavior for blocking API calls.
    """

    def __init__(self, enter: Callable[[], T_co]):
        self._enter = enter

    def __enter__(self) -> T_co:
        return self._enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # You can customize cleanup here if needed
        return False  # don't suppress exceptions


class AsyncResponseContextManager(Generic[T_co]):
    """
    An asynchronous context manager wrapper for API responses.

    Wraps an `AsyncContextManager[T]` and delegates enter/exit lifecycle.
    Useful for cleanly handling async resource lifetimes, like streamed HTTP responses.
    """

    def __init__(self, cm: AsyncContextManager[T_co]):
        self._cm = cm

    async def __aenter__(self) -> T_co:
        return await self._cm.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._cm.__aexit__(exc_type, exc_val, exc_tb)
