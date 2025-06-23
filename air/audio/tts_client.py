"""
Module providing tts client classes (both synchronous and asynchronous).

This module includes:
  - `AsyncTTSClient` for asynchronous calls.
  - `TTSClient` for synchronous calls.

All responses are validated using Pydantic models.
"""

from typing import Any

import aiohttp
import requests

from air.types.audio import TTSResponse


class AsyncTTSClient:  # pylint: disable=too-few-public-methods
    """
    An asynchronous client for the text-to-speech endpoint.

    This class handles sending requests to the text-to-speech endpoint
    and converts the responses into Pydantic models for type safety.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}

    async def create(  # pylint: disable=too-many-positional-arguments
        self,
        model: str,
        input: str,
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
            Raw binary audio data
        """
        endpoint = f"{self.base_url}/audio/speech"
        payload = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        if timeout is not None:
            payload["timeout"] = timeout
        if extra_body:
            payload.update(extra_body)

        # Start with built-in auth/JSON headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/octet-stream",
        }
        # Merge in default_headers
        headers.update(self.default_headers)
        # Merge in extra_headers (overwrites if collision)
        if extra_headers:
            headers.update(extra_headers)

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


class TTSClient:  # pylint: disable=too-few-public-methods
    """
    A synchronous client for the text-to-speech endpoint.

    This class handles sending requests to the text-to-speech endpoint
    and converts the responses into Pydantic models for type safety.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}

    def create(  # pylint: disable=too-many-positional-arguments
        self,
        model: str,
        input: str,
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
            Raw binary audio data
        """
        endpoint = f"{self.base_url}/audio/speech"
        payload = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        if timeout is not None:
            payload["timeout"] = timeout
        if extra_body:
            payload.update(extra_body)

        # Start with built-in auth/JSON headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/octet-stream",
        }
        # Merge in default_headers
        headers.update(self.default_headers)

        # Merge in extra_headers (overwrites if collision)
        if extra_headers:
            headers.update(extra_headers)

        resp = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=timeout if timeout is not None else 60,
        )
        resp.raise_for_status()
        return TTSResponse(resp.content)
