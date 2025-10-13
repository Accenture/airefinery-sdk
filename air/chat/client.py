"""
Module providing chat client classes (both synchronous and asynchronous)
that organize sub-clients (e.g., completions) under a `.completions` property.

This module includes:
  - ChatClient and ChatCompletionsClient for synchronous calls.
  - AsyncChatClient and AsyncChatCompletionsClient for asynchronous calls.

All clients call the "/v1/chat/completions" endpoint, and responses are
validated using Pydantic models.

Streaming behavior is strict by default:
- Non-streaming: returns ChatCompletion.
- Streaming: yields ChatCompletionChunk.
- Any malformed or error frames will raise SSEStreamError or ChunkValidationError.

Example usage:

    from air.chat.client import ChatClient, AsyncChatClient

    # Synchronous usage (non-stream):
    sync_client = ChatClient(
        base_url="https://api.airefinery.accenture.com",
        api_key="...",
        default_headers={"X-Client-Version": "1.2.3"},
    )
    resp = sync_client.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        messages=[...],
        timeout=10.0,
        extra_headers={"X-Request-Id": "abc-123"},
        extra_body={"input_type": "query"},
        stream=False,
    )

    # Asynchronous usage (stream):
    async_client = AsyncChatClient(
        base_url="https://api.airefinery.accenture.com",
        api_key="...",
        default_headers={"X-Client-Version": "1.2.3"},
    )
    async for chunk in await async_client.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        messages=[...],
        timeout=60.0,
        extra_headers={"X-Request-Id": "xyz-789"},
        extra_body={"input_type": "query"},
        stream=True,
    ):
        # chunk is a ChatCompletionChunk
        ...

Thread safety:
- Instances are stateless per request and safe to share in the same event loop or thread.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    Literal,
    Optional,
    Union,
    overload,
)

import aiohttp
import requests

from air import BASE_URL
from air.auth.token_provider import TokenProvider
from air.types import ChatCompletion, ChatCompletionChunk
from air.utils import get_base_headers, get_base_headers_async

logger = logging.getLogger(__name__)

ENDPOINT_COMPLETIONS = "{base_url}/v1/chat/completions"


class SSEStreamError(RuntimeError):
    """Raised when the upstream sends an explicit error event or malformed SSE frames."""

    pass


class ChunkValidationError(SSEStreamError):
    """Raised when an SSE JSON frame cannot be validated as a ChatCompletionChunk."""

    pass


def _build_payload(
    *,
    model: str,
    messages: list,
    extra_body: object | None,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct JSON payload for chat completions.

    Args:
        model: Target model to query.
        messages: Conversation messages (list of dicts).
        extra_body: Optional additional request body.
        kwargs: Extra parameters (e.g., temperature, top_p, max_tokens, stream).

    Returns:
        JSON-serializable payload dictionary.
    """
    return {
        "model": model,
        "messages": messages,
        "extra_body": extra_body,
        **kwargs,
    }


def _build_headers_sync(
    api_key: str | TokenProvider,
    default_headers: dict[str, str] | None,
    extra_headers: dict[str, str] | None,
    *,
    streaming: bool,
) -> dict[str, str]:
    """Build HTTP headers for the sync client.

    Args:
        api_key: API key or token provider.
        default_headers: Base headers applied to all requests of this client.
        extra_headers: Per-request headers.
        streaming: Whether this request uses SSE streaming.

    Returns:
        Headers dict ready for requests.post.
    """
    headers = get_base_headers(api_key)
    if default_headers:
        headers.update(default_headers)
    if extra_headers:
        headers.update(extra_headers)
    if streaming:
        headers.setdefault("Accept", "text/event-stream")
        headers.setdefault("Cache-Control", "no-cache")
    return headers


async def _build_headers_async(
    api_key: str | TokenProvider,
    default_headers: dict[str, str] | None,
    extra_headers: dict[str, str] | None,
    *,
    streaming: bool,
) -> dict[str, str]:
    """Build HTTP headers for the async client.

    Args:
        api_key: API key or token provider.
        default_headers: Base headers applied to all requests of this client.
        extra_headers: Per-request headers.
        streaming: Whether this request uses SSE streaming.

    Returns:
        Headers dict ready for aiohttp requests.
    """
    headers = await get_base_headers_async(api_key)
    if streaming:
        headers.update(
            {
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
            }
        )
    if default_headers:
        headers.update(default_headers)
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _parse_sse_frame_to_obj(frame: str) -> Dict[str, Any] | None:
    """Parse a single SSE frame into a JSON object.

    Handles:
    - Comments/heartbeats (lines starting with ":") — ignored.
    - "event:" and "data:" lines.
    - Multi-line data concatenation.
    - Proxy variations where a single JSON line lacks "data:".
    - "[DONE]" termination — returns None to signal end.

    Args:
        frame: Raw SSE frame text (CRLF normalized to LF elsewhere).

    Returns:
        Parsed JSON object, or None if the frame is a termination signal.

    Raises:
        SSEStreamError: On 'event: error' or irrecoverably malformed JSON.
    """
    if not frame.strip():
        return None

    event: Optional[str] = None
    data_lines: list[str] = []

    for raw_line in frame.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].lstrip() or None
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        # Some proxies strip 'data:' for single-line payloads
        data_lines.append(line)

    if not data_lines:
        return None

    data_content = "\n".join(data_lines).strip()

    if data_content == "[DONE]":
        return None

    if (event or "").lower() == "error":
        raise SSEStreamError(f"Upstream SSE error event: {data_content}")

    try:
        return json.loads(data_content)
    except json.JSONDecodeError:
        try:
            return json.loads(data_lines[0])
        except Exception as sse_err:
            raise SSEStreamError(
                f"Failed to parse SSE JSON frame: {data_lines[0]!r}"
            ) from sse_err


def _ensure_valid_chunk(obj: Dict[str, Any]) -> ChatCompletionChunk:
    """Validate a parsed SSE JSON object as ChatCompletionChunk.

    Args:
        obj: Parsed SSE JSON payload.

    Returns:
        A validated ChatCompletionChunk.

    Raises:
        SSEStreamError: If the payload is an explicit error envelope.
        ChunkValidationError: If validation to ChatCompletionChunk fails.
    """
    if obj.get("object") == "error" or "error" in obj:
        raise SSEStreamError(f"Upstream error payload: {obj}")
    try:
        return ChatCompletionChunk.model_validate(obj)
    except Exception as err:
        raise ChunkValidationError(
            f"Invalid chunk payload (strict streaming): {obj}"
        ) from err


def _stream_sync_chunks(
    resp: requests.Response,
) -> Generator[ChatCompletionChunk, None, None]:
    """Yield validated SSE chunks from a requests.Response (strict mode).

    Uses incremental decoding and frame splitting by blank lines.

    Args:
        resp: Response object opened with stream=True.

    Yields:
        ChatCompletionChunk objects.

    Raises:
        SSEStreamError: For upstream error frames or malformed JSON.
        ChunkValidationError: If a frame cannot be validated to ChatCompletionChunk.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""

    def _yield_frames_from_buffer() -> Generator[str, None, None]:
        nonlocal buffer
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            if not frame.strip():
                continue
            yield frame

    for chunk in resp.iter_content(chunk_size=8192):
        if not chunk:
            # Upstream ended (possibly without emitting [DONE])
            return

        text = decoder.decode(chunk)
        if text:
            text = text.replace("\r\n", "\n")
            buffer += text

        for frame in _yield_frames_from_buffer():
            obj = _parse_sse_frame_to_obj(frame)
            if obj is None:
                return
            yield _ensure_valid_chunk(obj)


async def _stream_async_chunks(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    json_body: Dict[str, Any],
    headers: Dict[str, str],
    *,
    idle_timeout_sec: float = 65.0,
    max_frame_bytes: int = 1 << 20,
    chunk_bytes: int = 8192,
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """Yield validated SSE chunks from an aiohttp response (strict mode).

    Features:
    - Incremental UTF-8 decoding.
    - CRLF normalization.
    - Idle timeout to prevent hangs.
    - Simple frame size protection.
    - "[DONE]" termination handling.

    Args:
        session: Active aiohttp ClientSession.
        method: HTTP method (e.g., "POST").
        url: Endpoint URL.
        json_body: JSON payload.
        headers: Request headers.
        idle_timeout_sec: Idle time without data before timeout.
        max_frame_bytes: Maximum allowed size per frame (bytes).
        chunk_bytes: Size of each read chunk from the network (bytes).

    Yields:
        ChatCompletionChunk objects.

    Raises:
        SSEStreamError: On upstream error frames or malformed JSON.
        ChunkValidationError: If a frame cannot be validated to ChatCompletionChunk.
        asyncio.TimeoutError: On idle timeout.
    """
    async with session.request(method, url, json=json_body, headers=headers) as resp:
        resp.raise_for_status()

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buffer = ""

        async def _yield_frames_from_buffer() -> AsyncGenerator[str, None]:
            nonlocal buffer
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                if not frame.strip():
                    continue
                yield frame

        try:
            while True:
                try:
                    async with asyncio.timeout(idle_timeout_sec):
                        chunk = await resp.content.read(chunk_bytes)
                except asyncio.TimeoutError:
                    raise asyncio.TimeoutError(
                        f"SSE idle timeout (> {idle_timeout_sec}s) without data"
                    ) from None

                if not chunk:
                    # Upstream ended (possibly without emitting [DONE])
                    return

                text = decoder.decode(chunk)
                if text:
                    text = text.replace("\r\n", "\n")
                    buffer += text

                async for frame in _yield_frames_from_buffer():
                    if len(frame.encode("utf-8", "ignore")) > max_frame_bytes:
                        raise SSEStreamError(
                            f"SSE frame exceeds {max_frame_bytes} bytes; aborting"
                        )

                    obj = _parse_sse_frame_to_obj(frame)
                    if obj is None:
                        return

                    yield _ensure_valid_chunk(obj)
        except asyncio.CancelledError:
            # Propagate cancellation to allow consumers to stop reading cleanly
            raise


class ChatCompletionsClient:  # pylint: disable=too-few-public-methods
    """Synchronous client for the chat completions endpoint."""

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str = BASE_URL,
        default_headers: dict[str, str] | None = None,
    ):
        """Initialize the synchronous completions client.

        Args:
            api_key: API key or token provider for authorization.
            base_url: Base URL of the API.
            default_headers: Headers to include in every request.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}

    @overload
    def create(
        self,
        *,
        model: str,
        messages: list,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        stream: Literal[False] = False,
        **kwargs,
    ) -> ChatCompletion: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: list,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        stream: Literal[True],
        **kwargs,
    ) -> Generator[ChatCompletionChunk, None, None]: ...

    def create(
        self,
        *,
        model: str,
        messages: list,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        **kwargs,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Create a chat completion synchronously.

        Supports both non-streaming and streaming modes. Streaming is strict.
        Malformed frames or explicit error frames raise SSEStreamError or
        ChunkValidationError.

        Args:
            model: Model name.
            messages: Conversation messages (list of dicts).
            timeout: Request timeout in seconds (defaults to 60 if not provided).
            extra_headers: Additional headers specific to this request.
            extra_body: Additional data to include in the request body.
            **kwargs: Extra parameters (temperature, top_p, max_tokens, stream, etc.).

        Returns:
            - If stream=False: ChatCompletion.
            - If stream=True: generator yielding ChatCompletionChunk.

        Raises:
            requests.RequestException: For network-related errors.
            SSEStreamError: For upstream error frames or malformed SSE frames.
            ChunkValidationError: If a frame cannot be validated to ChatCompletionChunk.
        """
        endpoint = ENDPOINT_COMPLETIONS.format(base_url=self.base_url)
        streaming = bool(kwargs.get("stream", False))
        payload = _build_payload(
            model=model, messages=messages, extra_body=extra_body, kwargs=kwargs
        )
        headers = _build_headers_sync(
            self.api_key, self.default_headers, extra_headers, streaming=streaming
        )

        resp = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=timeout if timeout is not None else 60,
            stream=streaming,
        )
        resp.raise_for_status()

        if streaming:
            return _stream_sync_chunks(resp)

        return ChatCompletion.model_validate(resp.json())


class ChatClient:  # pylint: disable=too-few-public-methods
    """Higher-level synchronous chat client that groups related API calls."""

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str = BASE_URL,
        default_headers: dict[str, str] | None = None,
    ):
        """Initialize the synchronous chat client.

        Args:
            api_key: API key or token provider for authorization.
            base_url: API base URL.
            default_headers: Headers applied to all requests in this client.
        """
        self.completions = ChatCompletionsClient(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
        )


class AsyncChatCompletionsClient:  # pylint: disable=too-few-public-methods
    """Asynchronous client for the chat completions endpoint."""

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str = BASE_URL,
        default_headers: dict[str, str] | None = None,
    ):
        """Initialize the asynchronous completions client.

        Args:
            api_key: API key or token provider for authorization.
            base_url: Base URL of the API.
            default_headers: Headers included in every request made by this client.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: list,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        stream: Literal[False] = False,
        **kwargs,
    ) -> ChatCompletion: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: list,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        stream: Literal[True],
        **kwargs,
    ) -> AsyncGenerator[ChatCompletionChunk, None]: ...

    async def create(
        self,
        *,
        model: str,
        messages: list,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        **kwargs,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Create a chat completion asynchronously.

        Supports both non-streaming and streaming modes. Streaming is strict.
        Malformed frames or explicit error frames raise SSEStreamError or
        ChunkValidationError.

        Args:
            model: Model name.
            messages: Conversation messages (list of dicts).
            timeout: Request timeout in seconds (defaults to 60 if not provided).
            extra_headers: Additional headers specific to this request.
            extra_body: Additional data to include in the request body.
            **kwargs: Extra parameters (temperature, top_p, max_tokens, stream, etc.).

        Returns:
            - If stream=False: ChatCompletion.
            - If stream=True: async generator yielding ChatCompletionChunk.

        Raises:
            aiohttp.ClientError: For network-related errors.
            SSEStreamError: For upstream error frames or malformed SSE frames.
            ChunkValidationError: If a frame cannot be validated to ChatCompletionChunk.
            asyncio.TimeoutError: If the SSE idle timeout is exceeded.
        """
        endpoint = ENDPOINT_COMPLETIONS.format(base_url=self.base_url)
        streaming = bool(kwargs.get("stream", False))
        payload = _build_payload(
            model=model, messages=messages, extra_body=extra_body, kwargs=kwargs
        )
        headers = await _build_headers_async(
            self.api_key, self.default_headers, extra_headers, streaming=streaming
        )

        if not streaming:
            client_timeout = aiohttp.ClientTimeout(total=timeout if timeout else 60)
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(
                    endpoint, json=payload, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    return ChatCompletion.model_validate(await resp.json())

        client_timeout = aiohttp.ClientTimeout(
            total=None if not timeout else timeout,
            sock_read=None if not timeout else timeout,
        )

        async def _agen() -> AsyncGenerator[ChatCompletionChunk, None]:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async for item in _stream_async_chunks(
                    session,
                    method="POST",
                    url=endpoint,
                    json_body=payload,
                    headers=headers,
                ):
                    yield item

        return _agen()


class AsyncChatClient:  # pylint: disable=too-few-public-methods
    """Higher-level asynchronous chat client that groups related API calls."""

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str = BASE_URL,
        default_headers: dict[str, str] | None = None,
    ):
        """Initialize the asynchronous chat client.

        Args:
            api_key: API key or token provider for authorization.
            base_url: API base URL.
            default_headers: Headers applied to all requests in this client.
        """
        self.completions = AsyncChatCompletionsClient(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
        )
