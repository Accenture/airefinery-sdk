"""
Module providing asr client classes (both synchronous and asynchronous).

This module includes:
  - `AsyncASRClient` for asynchronous calls.
  - `ASRClient` for synchronous calls.

All responses are validated using Pydantic models.
"""

import json
import os
from typing import IO, Union

import aiohttp
import requests

from air.types import ASRResponse
from air.types.audio import ChunkingStrategy


class AsyncASRClient:  # pylint: disable=too-few-public-methods
    """
    An asynchronous client for the speech-to-text endpoint.

    This class handles sending requests to the speech-to-text endpoint
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

    async def create(
        self,
        model: str,
        file: IO[bytes],
        chunking_strategy: Union[str, ChunkingStrategy] = "auto",
        language: str = "en-US",
        response_format: str = "json",
        stream: bool = False,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> ASRResponse:
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

        endpoint = f"{self.base_url}/audio/transcriptions"

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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
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


class ASRClient:  # pylint: disable=too-few-public-methods
    """
    A synchronous client for the speech-to-text endpoint.

    This class handles sending requests to the speech-to-text endpoint
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

    def create(
        self,
        model: str,
        file: IO[bytes],
        chunking_strategy: Union[str, ChunkingStrategy] = "auto",
        language: str = "en-US",
        response_format: str = "json",
        stream: bool = False,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> ASRResponse:
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

        endpoint = f"{self.base_url}/audio/transcriptions"
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        # Merge in default_headers
        headers.update(self.default_headers)

        # Merge in extra_headers (overwrites if collision)
        if extra_headers:
            headers.update(extra_headers)

        resp = requests.post(
            endpoint,
            data=payload,
            files=files_payload,
            headers=headers,
            timeout=timeout if timeout is not None else 60,
        )
        resp.raise_for_status()
        return ASRResponse.model_validate(resp.json())
