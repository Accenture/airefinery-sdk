"""
Module providing tts client classes (both synchronous and asynchronous).

This module includes:
  - `AsyncTTSClient` for asynchronous calls with batch and streaming support.
  - `TTSClient` for synchronous calls with batch and streaming support.
  - `StreamingResponse` for streaming audio responses.

All responses are validated using Pydantic models.
"""

import io
import logging
from functools import cached_property
from typing import Any, AsyncGenerator

import aiohttp
import requests

from air import BASE_URL, __version__
from air.auth.token_provider import TokenProvider
from air.types.audio import TTSResponse
from air.utils import get_base_headers, get_base_headers_async

logger = logging.getLogger(__name__)


ENDPOINT_SPEECH = "{base_url}/v1/audio/speech"


class StreamingResponse:
    """Wrapper to make streaming responses Open AI compatible.

    Provides both sync and async iteration over audio chunks, with methods
    for saving streamed content to files. Automatically handles the difference
    between sync and async streaming based on the is_async flag.

    Args:
        stream_generator: Generator or async generator for audio chunks.
        is_async: Whether this is an async streaming response.

    Usage:
        # Async streaming
        async with client.with_streaming_response.create(...) as response:
            async for chunk in response:
                process(chunk)

        # Sync streaming
        with client.with_streaming_response.create(...) as response:
            for chunk in response:
                process(chunk)
    """

    def __init__(self, stream_generator, is_async=True):
        """Initialize streaming response wrapper.

        Args:
            stream_generator: Generator or async generator for audio chunks.
            is_async: Whether this is an async streaming response.
        """
        self._stream_generator = stream_generator
        self._chunks = None
        self._is_async = is_async

    def __enter__(self):
        """Enter sync context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit sync context manager."""
        pass

    def __iter__(self):
        """Make sync StreamingResponse iterable.

        Raises:
            TypeError: If used with async StreamingResponse.
        """
        if self._is_async:
            raise TypeError("Use 'async for' with async StreamingResponse")
        return iter(self._stream_generator())

    def __aiter__(self):
        """Make async StreamingResponse async iterable

        Raises:
            TypeError: If used with sync StreamingResponse.
        """
        if not self._is_async:
            raise TypeError("Use 'for' with sync StreamingResponse")
        return self._stream_generator.__aiter__()

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        pass

    async def _collect_chunks_async(self):
        """Collect all chunks when needed (async version).

        Returns:
            bytes: Complete audio data from all chunks.
        """
        if self._chunks is None:
            buffer = io.BytesIO()
            async for chunk in self._stream_generator:
                buffer.write(chunk)
            self._chunks = buffer.getvalue()
        return self._chunks

    def _collect_chunks_sync(self):
        """Collect all chunks when needed (sync version).

        Returns:
            bytes: Complete audio data from all chunks.
        """
        if self._chunks is None:
            buffer = io.BytesIO()
            for chunk in self._stream_generator():
                buffer.write(chunk)
            self._chunks = buffer.getvalue()
        return self._chunks

    async def stream_to_file(self, file_path):
        """OpenAI-compatible stream_to_file method.

        Args:
            file_path: Path where to save the audio file.
        """
        if self._is_async:
            content = await self._collect_chunks_async()
        else:
            content = self._collect_chunks_sync()
        with open(file_path, "wb") as f:
            f.write(content)


class AsyncTTSClient:  # pylint: disable=too-few-public-methods
    """
    An asynchronous client for the text-to-speech endpoint.

    This class handles sending requests to the text-to-speech endpoint
    and converts the responses into TTSResponse objects for type safety.

    Supports both batch and streaming synthesis:
    - create(): Returns complete audio (batch mode)
    - with_streaming_response.create(): Yields audio chunks (streaming mode)
    """

    endpoint_speech = "{base_url}/audio/speech"

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str = BASE_URL,
        default_headers: dict[str, str] | None = None,
    ):
        """Initialize the TTS client.

        Args:
            api_key (str | TokenProvider): Credential that will be placed in the
                ``Authorization`` header of every request.
                * **str** – a raw bearer token / API key.
                * **TokenProvider** – an object whose ``token()`` (and
                  ``token_async()``) method returns a valid bearer token.  The
                  client calls the provider automatically before each request.

            base_url: Base URL for TTS.
            default_headers: Default headers to include in requests.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}

    async def create(
        self,
        model: str,
        input: str,
        *,
        voice: str,
        response_format: str = "mp3",  # Optional with default
        speed: float = 1.0,  # Optional with default
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> TTSResponse:
        """
        Creates text-to-speech conversion asynchronously.

        Args:
            model: The model identifier for TTS
            input: Text string to convert to speech
            voice: Voice name (e.g., "en-US-JennyNeural")
            response_format: Audio format (mp3, wav, etc.)
            speed: Speech speed (0.25 to 4.0, default 1.0)
            timeout: Request timeout in seconds
            extra_headers: Additional HTTP headers to include
            extra_body: Additional JSON properties to include in the request body

        Returns:
            TTSResponse with complete audio data

        Raises:
            aiohttp.ClientError: If network request fails.
        """
        endpoint = ENDPOINT_SPEECH.format(base_url=self.base_url)
        payload = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
            "stream": False,  # Batch mode: server returns complete audio at once
        }
        if timeout is not None:
            payload["timeout"] = timeout
        if extra_body:
            payload.update(extra_body)

        # Start with built-in auth/JSON headers
        headers = await get_base_headers_async(
            self.api_key, extra_headers={"Accept": "application/octet-stream"}
        )
        # Merge in default_headers
        headers.update(self.default_headers)
        # Merge in extra_headers (overwrites if collision)
        if extra_headers:
            headers.update(extra_headers)

        # Batch mode - return complete audio
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(
                        total=timeout if timeout is not None else 60
                    ),
                ) as resp:
                    resp.raise_for_status()
                    return TTSResponse(await resp.read())
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {e}")
            raise

    @cached_property
    def with_streaming_response(self) -> "AsyncTTSClientWithStreamingResponse":
        """
        Access streaming response methods.

        This property provides OpenAI-compatible streaming syntax:
        client.audio.speech.with_streaming_response.create(...)

        Returns:
            AsyncTTSClientWithStreamingResponse: Wrapper for streaming methods
        """
        return AsyncTTSClientWithStreamingResponse(self)


class AsyncTTSClientWithStreamingResponse:
    """
    Streaming response wrapper for AsyncTTSClient.

    This wrapper enables OpenAI-compatible streaming syntax using
    client.audio.speech.with_streaming_response.create(...)
    """

    def __init__(self, client: AsyncTTSClient):
        """
        Initialize the streaming wrapper.

        Args:
            client: The AsyncTTSClient instance to wrap
        """
        self._client = client

    async def create(
        self,
        model: str,
        input: str,
        *,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        timeout: float | None = None,
        chunk_size: int = 1024,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> "StreamingResponse":
        """
        Creates streaming text-to-speech conversion asynchronously.

        Args:
            model: The model identifier for TTS
            input: Text string to convert to speech
            voice: Voice name (e.g., "en-US-JennyNeural")
            response_format: Audio format (mp3, wav, etc.)
            speed: Speech speed (0.25 to 4.0, default 1.0)
            timeout: Request timeout in seconds
            chunk_size: Bytes per chunk when streaming (default: 1024)
            extra_headers: Additional HTTP headers to include
            extra_body: Additional JSON properties to include in the request body

        Returns:
            StreamingResponse

        Raises:
            aiohttp.ClientError: If network request fails.
        """
        endpoint = ENDPOINT_SPEECH.format(base_url=self._client.base_url)
        payload = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
            "stream": True,  # Streaming mode: server sends audio in chunks as generated
        }
        if timeout is not None:
            payload["timeout"] = timeout
        if extra_body:
            payload.update(extra_body)

        # Start with built-in auth/JSON headers
        headers = get_base_headers(self._client.api_key)

        # Merge in default_headers
        headers.update(self._client.default_headers)
        # Merge in extra_headers (overwrites if collision)
        if extra_headers:
            headers.update(extra_headers)

        # Streaming mode - return async generator
        async def _stream_generator() -> AsyncGenerator[bytes, None]:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(
                            total=timeout if timeout is not None else 60
                        ),
                    ) as resp:
                        resp.raise_for_status()

                        async for chunk in resp.content.iter_chunked(chunk_size):
                            if chunk:  # Filter out keep-alive chunks
                                yield chunk
            except aiohttp.ClientError as e:
                logger.error(f"Network error: {e}")
                raise

        return StreamingResponse(_stream_generator(), is_async=True)


class TTSClient:  # pylint: disable=too-few-public-methods
    """
    A synchronous client for the text-to-speech endpoint.

    This class handles sending requests to the text-to-speech endpoint
    and converts the responses into TTSResponse objects for type safety.

    Supports both batch and streaming synthesis:
    - create(): Returns complete audio (batch mode)
    - with_streaming_response.create(): Yields audio chunks (streaming mode)
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

    def create(
        self,
        model: str,
        input: str,
        *,
        voice: str,
        response_format: str = "mp3",  # Optional with default
        speed: float = 1.0,  # Optional with default
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> TTSResponse:
        """
        Creates text-to-speech conversions synchronously.

        Args:
            model: The model identifier for TTS
            input: Text string to convert to speech
            voice: Voice name (e.g., "en-US-JennyNeural")
            response_format: Audio format (mp3, wav, etc.)
            speed: Speech speed (0.25 to 4.0, default 1.0)
            timeout: Request timeout in seconds
            extra_headers: Additional HTTP headers to include
            extra_body: Additional JSON properties to include in the request body

        Returns:
            TTSResponse with complete audio data

        Raises:
            requests.RequestException: If network request fails.
        """
        endpoint = ENDPOINT_SPEECH.format(base_url=self.base_url)
        payload = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
            "stream": False,  # Batch mode: server returns complete audio at once
        }
        if timeout is not None:
            payload["timeout"] = timeout
        if extra_body:
            payload.update(extra_body)

        # Start with built-in auth/JSON headers
        headers = get_base_headers(
            self.api_key, extra_headers={"Accept": "application/octet-stream"}
        )
        # Merge in default_headers
        headers.update(self.default_headers)

        # Merge in extra_headers (overwrites if collision)
        if extra_headers:
            headers.update(extra_headers)

        try:
            resp = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=timeout if timeout is not None else 60,
            )
            resp.raise_for_status()
            return TTSResponse(resp.content)
        except requests.RequestException as e:
            logger.error(f"Network error: {e}")
            raise

    @cached_property
    def with_streaming_response(self) -> "TTSClientWithStreamingResponse":
        """
        Access streaming response methods.

        This property provides OpenAI-compatible streaming syntax:
        client.audio.speech.with_streaming_response.create(...)

        Returns:
            TTSClientWithStreamingResponse: Wrapper for streaming methods
        """
        return TTSClientWithStreamingResponse(self)


class TTSClientWithStreamingResponse:
    """
    Streaming response wrapper for TTSClient.

    This wrapper enables OpenAI-compatible streaming syntax using
    client.audio.speech.with_streaming_response.create(...)
    """

    def __init__(self, client: TTSClient):
        """
        Initialize the streaming wrapper.

        Args:
            client: The TTSClient instance to wrap
        """
        self._client = client

    def create(
        self,
        model: str,
        input: str,
        *,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        timeout: float | None = None,
        chunk_size: int = 1024,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> "StreamingResponse":
        """
        Creates streaming text-to-speech conversion synchronously.

        Args:
            model: The model identifier for TTS
            input: Text string to convert to speech
            voice: Voice name (e.g., "en-US-JennyNeural")
            response_format: Audio format (mp3, wav, etc.)
            speed: Speech speed (0.25 to 4.0, default 1.0)
            timeout: Request timeout in seconds
            chunk_size: Bytes per chunk when streaming (default: 1024)
            extra_headers: Additional HTTP headers to include
            extra_body: Additional JSON properties to include in the request body

        Returns:
            StreamingResponse

        Raises:
            requests.RequestException: If network request fails.
        """
        endpoint = ENDPOINT_SPEECH.format(base_url=self._client.base_url)
        payload = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
            "stream": True,  # Streaming mode: server sends audio in chunks as generated
        }
        if timeout is not None:
            payload["timeout"] = timeout
        if extra_body:
            payload.update(extra_body)

        # Start with built-in auth/JSON headers
        headers = get_base_headers(self._client.api_key)

        # Merge in default_headers
        headers.update(self._client.default_headers)
        # Merge in extra_headers (overwrites if collision)
        if extra_headers:
            headers.update(extra_headers)

        # Streaming mode - return iterator
        def _stream_iterator():
            """Generate audio chunks synchronously."""

            try:
                resp = requests.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=timeout if timeout is not None else 60,
                    stream=True,
                )
                resp.raise_for_status()

                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        yield chunk
            except requests.RequestException as e:
                logger.error(f"Network error: {e}")
                raise

        return StreamingResponse(_stream_iterator, is_async=False)
