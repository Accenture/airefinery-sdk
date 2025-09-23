"""
Module providing asr client classes (both synchronous and asynchronous).

This module includes:
  - `AsyncASRClient` for asynchronous calls.
  - `ASRClient` for synchronous calls.

All responses are validated using Pydantic models.
"""

import json
import os
from contextlib import asynccontextmanager, contextmanager
from functools import cached_property
from typing import (
    IO,
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterator,
    Literal,
    Union,
    cast,
    overload,
)

import aiohttp
import requests
from pydantic import TypeAdapter

from air import BASE_URL, __version__
from air.auth.token_provider import TokenProvider
from air.types import ASRResponse
from air.types.audio import (
    APIResponse,
    AsyncAPIResponse,
    AsyncResponseContextManager,
    AsyncStream,
    ChunkingStrategy,
    ResponseContextManager,
    Stream,
    TranscriptionStreamEvent,
)
from air.utils import get_base_headers, get_base_headers_async

ENDPOINT_TRANSCRIPTIONS = "{base_url}/v1/audio/transcriptions"


class AsyncASRClient:  # pylint: disable=too-few-public-methods
    """
    An asynchronous client for the speech-to-text endpoint.

    This class handles sending requests to the speech-to-text endpoint
    and converts the responses into Pydantic models for type safety.
    """

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str = BASE_URL,
        default_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}

    @cached_property
    def with_streaming_response(self) -> "AsyncTranscriptionsWithStreamingResponse":
        """
        Returns a context manager for asynchronous transcriptions with real-time streaming responses.
        """
        return AsyncTranscriptionsWithStreamingResponse(self)

    @overload
    async def create(
        self, *, stream: Literal[True], **kwargs
    ) -> AsyncStream[TranscriptionStreamEvent]: ...

    @overload
    async def create(
        self, *, stream: Literal[False] = False, **kwargs
    ) -> ASRResponse: ...

    async def create(
        self,
        model: str,
        file: IO[bytes],
        *,
        chunking_strategy: Union[str, ChunkingStrategy] = "auto",
        language: str = "en-US",
        response_format: str = "json",
        stream: bool = False,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> (
        ASRResponse | AsyncStream[TranscriptionStreamEvent]
    ):  # pylint: disable=too-many-arguments, disable=too-many-locals
        """
        Creates speech-to-text transcriptions asynchronously.

        Args:
            model: The model identifier for ASR
            file: List of audio file paths to transcribe
            chunking_strategy: Optional Parameters to configure server-side VAD
            language: Optional override for the speech recognition language.
            response_format: Output format (currently unused, reserved for future use).
            stream: Whether streaming is enabled (currently unused).
            timeout: Request timeout in seconds
            extra_headers: Additional HTTP headers to include
            extra_body: Additional parameters or overides of listed parameters
            **kwargs: Additional parameters to pass to the API (e.g., language)

        Returns:
            ASRResponse: The response containing transcription results
        """

        endpoint = ENDPOINT_TRANSCRIPTIONS.format(base_url=self.base_url)

        # build payload
        form = aiohttp.FormData()
        form.add_field("model", model)  # ordinary field
        form.add_field(
            "chunking_strategy",
            (
                json.dumps(chunking_strategy)
                if isinstance(chunking_strategy, dict)
                else "auto"
            ),
        )
        form.add_field("language", language)
        form.add_field("response_format", response_format)
        form.add_field("stream", str(stream))
        form.add_field("timeout", str(timeout))
        form.add_field(
            "extra_body", json.dumps(extra_body) if extra_body is not None else "{}"
        )

        # Ensure the pointer is at the start
        file.seek(0)

        filename = os.path.basename(getattr(file, "name", "file.wav"))
        form.add_field(
            "file",  # field name must match FastAPI’s arg
            file,  # the binary file object
            filename=filename,
            content_type="audio/wav",  # only sending .wav audio files
        )

        # add any extra scalar params
        for k, v in kwargs.items():
            form.add_field(k, str(v))

        # Start with built-in auth/JSON headers
        headers = await get_base_headers_async(self.api_key, content_type=None)

        # Merge in default_headers
        headers.update(self.default_headers)
        # Merge in extra_headers (overwrites if collision)
        if extra_headers:
            headers.update(extra_headers)

        timeout_obj = (
            aiohttp.ClientTimeout(total=timeout)
            if timeout is not None
            else aiohttp.ClientTimeout(total=60)
        )

        if not stream:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    data=form,
                    headers=headers,
                    timeout=timeout_obj,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return ASRResponse.model_validate(data)

        class _StreamWrapper:
            def __aiter__(self) -> AsyncIterator[TranscriptionStreamEvent]:
                async def _stream():
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            endpoint,
                            data=form,
                            headers=headers,
                            timeout=timeout_obj,
                        ) as resp:
                            resp.raise_for_status()

                            async for raw in resp.content:
                                line = raw.decode().strip()
                                if line.startswith("data: [DONE]"):
                                    break
                                if not line.startswith("data:"):
                                    continue  # skip comments / blanks
                                try:
                                    event = TypeAdapter(
                                        TranscriptionStreamEvent
                                    ).validate_python(json.loads(line[5:].strip()))
                                    yield event
                                except Exception as e:
                                    print("invalid payload:", e)
                                    continue

                return _stream()

        return _StreamWrapper()


class AsyncTranscriptionsWithStreamingResponse:  # pylint: disable=too-few-public-methods
    """
    A wrapper class for creating asynchronous speech-to-text transcriptions
    with support for streaming responses.

    This class applies the same transcription logic to as `AsyncASRClient`,
    while returning responses using an async context manager.
    """

    def __init__(self, transcriptions: AsyncASRClient) -> None:
        self._transcriptions = transcriptions

    def create(
        self,
        model: str,
        file: IO[bytes],
        *,
        chunking_strategy: Union[str, ChunkingStrategy] = "auto",
        language: str = "en-US",
        response_format: str = "json",
        stream: bool = False,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> (
        AsyncResponseContextManager[AsyncAPIResponse[ASRResponse]]
        | AsyncResponseContextManager[
            AsyncAPIResponse[AsyncStream[TranscriptionStreamEvent]]
        ]
    ):  # pylint: disable=too-many-arguments, disable=too-many-locals
        """
        Creates speech-to-text transcriptions asynchronously with a streaming response.

        Args:
            model: The model identifier for ASR
            file: List of audio file paths to transcribe
            chunking_strategy: Optional Parameters to configure server-side VAD
            language: Optional override for the speech recognition language.
            response_format: Output format (currently unused, reserved for future use).
            stream: Whether streaming is enabled (currently unused).
            timeout: Request timeout in seconds
            extra_headers: Additional HTTP headers to include
            extra_body: Additional parameters or overides of listed parameters
            **kwargs: Additional parameters to pass to the API (e.g., language)

        Returns:
            An async context manager that yields either:
              - AsyncAPIResponse[ASRResponse] for non-streaming mode.
              - AsyncAPIResponse[AsyncStream[TranscriptionStreamEvent]] for streaming mode.
        """

        endpoint = ENDPOINT_TRANSCRIPTIONS.format(
            base_url=self._transcriptions.base_url
        )

        # build payload
        form = aiohttp.FormData()
        form.add_field("model", model)  # ordinary field
        form.add_field(
            "chunking_strategy",
            (
                json.dumps(chunking_strategy)
                if isinstance(chunking_strategy, dict)
                else "auto"
            ),
        )
        form.add_field("language", language)
        form.add_field("response_format", response_format)
        form.add_field("stream", str(stream))
        form.add_field("timeout", str(timeout))
        form.add_field(
            "extra_body", json.dumps(extra_body) if extra_body is not None else "{}"
        )

        # Ensure the pointer is at the start
        file.seek(0)

        filename = os.path.basename(getattr(file, "name", "file.wav"))
        form.add_field(
            "file",  # field name must match FastAPI’s arg
            file,  # the binary file object
            filename=filename,
            content_type="audio/wav",  # only sending .wav audio files
        )

        # add any extra scalar params
        for k, v in kwargs.items():
            form.add_field(k, str(v))

        # Start with built-in auth/JSON headers
        headers = get_base_headers(self._transcriptions.api_key, content_type=None)

        # Merge in default_headers
        headers.update(self._transcriptions.default_headers)
        # Merge in extra_headers (overwrites if collision)
        if extra_headers:
            headers.update(extra_headers)

        timeout_obj = (
            aiohttp.ClientTimeout(total=timeout)
            if timeout is not None
            else aiohttp.ClientTimeout(total=60)
        )
        if not stream:

            @asynccontextmanager
            async def _cm() -> AsyncGenerator[AsyncAPIResponse[ASRResponse], None]:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint, data=form, headers=headers, timeout=timeout_obj
                    ) as resp:
                        resp.raise_for_status()
                        parsed = ASRResponse.model_validate(await resp.json())
                        yield AsyncAPIResponse(
                            http_response=resp,
                            parsed=parsed,
                        )

            return AsyncResponseContextManager(_cm())

        async def _event_iter(
            resp: aiohttp.ClientResponse,
        ) -> AsyncIterator[str]:
            async for raw in resp.content:
                line = raw.decode().strip()
                yield line

        @asynccontextmanager
        async def _cm_stream() -> (
            AsyncGenerator[
                AsyncAPIResponse[AsyncStream[TranscriptionStreamEvent]], None
            ]
        ):
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint, data=form, headers=headers, timeout=timeout_obj
                ) as resp:
                    resp.raise_for_status()

                    stream_obj = cast(
                        AsyncStream[TranscriptionStreamEvent],
                        _event_iter(resp),
                    )
                    yield AsyncAPIResponse(
                        http_response=resp,
                        parsed=stream_obj,
                    )

        return AsyncResponseContextManager(_cm_stream())


class ASRClient:  # pylint: disable=too-few-public-methods
    """
    A synchronous client for the speech-to-text endpoint.

    This class handles sending requests to the speech-to-text endpoint
    and converts the responses into Pydantic models for type safety.
    """

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str,
        default_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}

    @cached_property
    def with_streaming_response(self) -> "TranscriptionsWithStreamingResponse":
        """
        Returns a context manager for synchronous transcriptions with real-time streaming responses.
        """
        return TranscriptionsWithStreamingResponse(self)

    @overload
    def create(
        self, *, stream: Literal[True], **kwargs
    ) -> Stream[TranscriptionStreamEvent]: ...

    @overload
    def create(self, *, stream: Literal[False] = False, **kwargs) -> ASRResponse: ...

    def create(
        self,
        model: str,
        file: IO[bytes],
        *,
        chunking_strategy: Union[str, ChunkingStrategy] = "auto",
        language: str = "en-US",
        response_format: str = "json",
        stream: bool = False,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> (
        ASRResponse | Stream[TranscriptionStreamEvent]
    ):  # pylint: disable=too-many-arguments, disable=too-many-locals
        """
        Creates speech-to-text transcriptions synchronously.

        Args:
            model: The model identifier for ASR
            file: List of audio file paths to transcribe
            timeout: Request timeout in seconds
            extra_headers: Additional HTTP headers to include
            **kwargs: Additional parameters to pass to the API (e.g., language)

        Returns:
            ASRResponse: The response containing transcription results
        """

        endpoint = ENDPOINT_TRANSCRIPTIONS.format(base_url=self.base_url)

        payload = {
            "model": model,
            "chunking_strategy": (
                json.dumps(chunking_strategy)
                if isinstance(chunking_strategy, dict)
                else "auto"
            ),
            "language": language,
            "response_format": response_format,
            "stream": stream,
            "extra_body": json.dumps(extra_body) if extra_body is not None else "{}",
            "timeout": timeout,
            **kwargs,
        }

        # file payload
        files_payload = []

        file.seek(0)
        filename = os.path.basename(getattr(file, "name", "file_.wav"))
        files_payload.append(
            (
                "file",  # field name
                (filename, file, "audio/wav"),  # (filename, fileobj, MIME)
            )
        )

        # Start with built-in auth/JSON headers
        headers = get_base_headers(self.api_key, content_type=None)

        # Merge in default_headers
        headers.update(self.default_headers)

        # Merge in extra_headers (overwrites if collision)
        if extra_headers:
            headers.update(extra_headers)

        if not stream:  # ---------- one-shot branch ------------
            resp = requests.post(
                endpoint,
                data=payload,
                files=files_payload,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            return ASRResponse.model_validate(resp.json())

        def _iter() -> Stream[TranscriptionStreamEvent]:
            with requests.post(
                endpoint,
                data=payload,
                files=files_payload,
                headers=headers,
                timeout=timeout,
                stream=True,  # << keep socket open
            ) as resp:
                resp.raise_for_status()

                for raw in resp.iter_lines(decode_unicode=True):
                    line = str(raw).strip()
                    if line.startswith("data: [DONE]"):
                        break
                    if not line.startswith("data:"):
                        continue  # skip comments / blanks
                    try:
                        event = TypeAdapter(TranscriptionStreamEvent).validate_python(
                            json.loads(line[5:].strip())
                        )
                        yield event
                    except Exception as e:
                        print("invalid payload:", e)
                        continue

        return _iter()


class TranscriptionsWithStreamingResponse:  # pylint: disable=too-few-public-methods
    """
    A wrapper class for creating synchronous speech-to-text transcriptions
    with support for streaming responses.

    This class applies the same transcription logic to as `ASRClient`,
    while returning responses using an async context manager.
    """

    def __init__(self, transcriptions: ASRClient) -> None:
        self._transcriptions = transcriptions

    def create(
        self,
        model: str,
        file: IO[bytes],
        *,
        chunking_strategy: Union[str, ChunkingStrategy] = "auto",
        language: str = "en-US",
        response_format: str = "json",
        stream: bool = False,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> (
        ResponseContextManager[APIResponse[ASRResponse]]
        | ResponseContextManager[APIResponse[Stream[TranscriptionStreamEvent]]]
    ):  # pylint: disable=too-many-arguments, disable=too-many-locals
        """
        Creates speech-to-text transcriptions synchronously with a streaming response.

        Args:
            model: The model identifier for ASR
            file: List of audio file paths to transcribe
            chunking_strategy: Optional Parameters to configure server-side VAD
            language: Optional override for the speech recognition language.
            response_format: Output format (currently unused, reserved for future use).
            stream: Whether streaming is enabled (currently unused).
            timeout: Request timeout in seconds
            extra_headers: Additional HTTP headers to include
            extra_body: Additional parameters or overides of listed parameters
            **kwargs: Additional parameters to pass to the API (e.g., language)

        Returns:
            An sync context manager that yields either:
              - APIResponse[ASRResponse] for non-streaming mode.
              - APIResponse[Stream[TranscriptionStreamEvent]] for streaming mode.
        """
        endpoint = ENDPOINT_TRANSCRIPTIONS.format(
            base_url=self._transcriptions.base_url
        )

        payload = {
            "model": model,
            "chunking_strategy": (
                json.dumps(chunking_strategy)
                if isinstance(chunking_strategy, dict)
                else "auto"
            ),
            "language": language,
            "response_format": response_format,
            "stream": stream,
            "extra_body": json.dumps(extra_body) if extra_body is not None else "{}",
            "timeout": timeout,
            **kwargs,
        }

        # file payload
        files_payload = []

        file.seek(0)
        filename = os.path.basename(getattr(file, "name", "file_.wav"))
        files_payload.append(
            (
                "file",  # field name
                (filename, file, "audio/wav"),  # (filename, fileobj, MIME)
            )
        )

        # Start with built-in auth/JSON headers
        headers = get_base_headers(self._transcriptions.api_key, content_type=None)

        # Merge in default_headers
        headers.update(self._transcriptions.default_headers)

        # Merge in extra_headers (overwrites if collision)
        if extra_headers:
            headers.update(extra_headers)

        if not stream:

            @contextmanager
            def _cm() -> Generator[APIResponse[ASRResponse], None, None]:
                with requests.post(
                    endpoint,
                    data=payload,
                    files=files_payload,
                    headers=headers,
                    timeout=timeout,
                ) as resp:
                    resp.raise_for_status()
                    parsed = ASRResponse.model_validate(resp.json())
                    yield APIResponse(
                        http_response=resp,
                        parsed=parsed,
                    )

            return ResponseContextManager(_cm().__enter__)

        def _event_iter(
            resp: requests.Response,
        ) -> Iterator[TranscriptionStreamEvent]:
            for raw in resp.iter_lines():
                line = raw.decode().strip()
                yield line

        @contextmanager
        def _cm_stream() -> (
            Generator[APIResponse[Stream[TranscriptionStreamEvent]], None, None]
        ):
            with requests.post(
                endpoint,
                data=payload,
                files=files_payload,
                headers=headers,
                timeout=timeout,
                stream=True,  # << keep socket open
            ) as resp:
                resp.raise_for_status()

                stream_obj = cast(
                    Stream[TranscriptionStreamEvent],
                    _event_iter(resp),
                )

                yield APIResponse(
                    http_response=resp,
                    parsed=stream_obj,
                )

        return ResponseContextManager(_cm_stream().__enter__)
