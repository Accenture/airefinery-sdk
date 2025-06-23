"""

Pydantic models for Automatic Speech Recognition (ASR) and Text-To-Speech (TTS) responses.
Response types for TTS audio operations.

This module provides TTSResponse which mimics OpenAI's HttpxBinaryResponseContent
for full compatibility with OpenAI's client interface.
This module provides:
  • TTSResponse         : TTS response containing audio bytes and text encoding
  • ASRResponse         :  ASR response containing multiple results
  • ChunkingStrategy    : Server VAD Configuration

"""

import json
import os
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Literal,
    Optional,
    Required,
    TypedDict,
    Union,
)

from air.types.base import CustomBaseModel


class TTSResponse:
    """
    Response wrapper for TTS audio data.
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

    def json(self, **kwargs: Any) -> Any:
        """Parse content as JSON (not applicable for audio, but maintaining interface)."""

        return json.loads(self.text, **kwargs)

    def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
        """Iterate over content in byte chunks."""
        if chunk_size is None:
            chunk_size = 1024

        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def iter_text(self, chunk_size: int | None = None) -> Iterator[str]:
        """Iterate over content as text chunks."""
        for chunk in self.iter_bytes(chunk_size):
            yield chunk.decode(self._encoding)

    def iter_lines(self) -> Iterator[str]:
        """Iterate over content line by line."""
        return iter(self.text.splitlines())

    def iter_raw(self, chunk_size: int | None = None) -> Iterator[bytes]:
        """Iterate over raw content in chunks."""
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

    async def aiter_bytes(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        """Async iteration over content in byte chunks."""
        if chunk_size is None:
            chunk_size = 1024

        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    async def aiter_text(self, chunk_size: int | None = None) -> AsyncIterator[str]:
        """Async iteration over content as text chunks."""
        async for chunk in self.aiter_bytes(chunk_size):
            yield chunk.decode(self._encoding)

    async def aiter_lines(self) -> AsyncIterator[str]:
        """Async iteration over content line by line."""
        for line in self.text.splitlines():
            yield line

    async def aiter_raw(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        """Async iteration over raw content in chunks."""
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
        self.stream_to_file(file, chunk_size=chunk_size)

    async def aclose(self) -> None:
        """Async close the response"""
        pass


class ASRResponse(CustomBaseModel):
    """Top-level Automatic Speech Recognition response returned by the API.

    Attributes:
        text: The transcription of the audio file
        success: Whether the transcription request was successful
        error: Optional error message if transcription was not successful
        confidence: Optional confidence of the tokens
    """

    text: Union[str, None]
    success: bool
    error: Optional[str] = None
    confidence: Optional[float] = None


class ChunkingStrategy(TypedDict, total=False):
    """
    Controls how the audio is cut into chunks.

    Attributes:
        type : Literal["server_vad"]: Selects server-side VAD chunking (required).
        prefix_padding_ms : int, optional:  Lead-in context before speech, 0–5000 ms.
        silence_duration_ms : int, optional: Trailing silence that closes a chunk, 0–5000 ms.
        threshold : float, optional: VAD sensitivity, 0.0–1.0 (currently ignored).
    """

    type: Required[Literal["server_vad"]]
    prefix_padding_ms: int  # initial_vad_delay, 0 - 5000 ms
    silence_duration_ms: int  # segmentation_vad_delay, 0 - 5000 ms
    threshold: float  # vad_sensitivity, 0.0 - 1.0 : NotImplemented
