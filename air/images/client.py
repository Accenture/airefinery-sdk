"""
Module providing clients for image operations.
All responses are validated using Pydantic models.

This module includes:
  - `ImagesClient` for synchronous calls.
  - `AsyncImagesClient` for asynchronous calls.

All clients call the `/images/generations` endpoint, and all responses
are validated using Pydantic models (`ImagesResponse`).
"""

import aiohttp
import requests

from air.types import ImagesResponse


class ImagesClient:
    """
    A synchronous client for the image endpoint.

    This class handles sending requests to the image endpoint
    and converts the responses into Pydantic models for type safety.
    """

    def __init__(
        self, base_url: str, api_key: str, default_headers: dict[str, str] | None = None
    ):
        """
        Initializes the synchronous image client.

        Args:
            base_url (str): Base URL of the API (e.g., "https://api.airefinery.accenture.com").
            api_key (str): API key for authorization.
            default_headers (dict[str, str] | None): Optional headers applied to every request.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        **kwargs,
    ) -> ImagesResponse:
        """
        Generates an image synchronously.

        Sends a POST to the `/images/generations` endpoint with the given prompt
        and model, and returns the parsed Pydantic response.

        Args:
            prompt (str): The text prompt guiding image generation.
            model (str): The model name (e.g., "dall-e-2", "stable-diffusion-v1").
            timeout (float | None): Max time (in seconds) to wait for a response.
                Defaults to 60 seconds if not provided.
            extra_headers (dict[str, str] | None): Request-specific headers
                that override any default headers.
            extra_body (object | None): Additional data to include in the
                request body, if needed.
            **kwargs: Additional generation parameters (e.g., "n", "size", "user").

        Returns:
            ImagesResponse: The parsed Pydantic model containing
                generated image URLs and metadata.
        """
        effective_timeout = timeout if timeout is not None else 60

        endpoint = f"{self.base_url}/images/generations"

        payload = {
            "model": model,
            "prompt": prompt,
            "timeout": effective_timeout,
            "extra_body": extra_body,
            **kwargs,
        }

        # Base authorization and JSON headers.
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Merge in default headers
        headers.update(self.default_headers)
        # Merge in request-specific headers last, overwriting if a key collides
        if extra_headers:
            headers.update(extra_headers)

        response = requests.post(
            endpoint, json=payload, headers=headers, timeout=effective_timeout
        )
        response.raise_for_status()
        return ImagesResponse.model_validate(response.json())


class AsyncImagesClient:
    """
    An asynchronous client for the image endpoint.

    This class handles sending requests to the image endpoint
    and converts the responses into Pydantic models for type safety.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_headers: dict[str, str] | None = None,
    ):
        """
        Initializes the asynchronous image client.

        Args:
            base_url (str): Base URL of the API (e.g., "https://api.airefinery.accenture.com").
            api_key (str): API key for authorization.
            default_headers (dict[str, str] | None): Optional headers applied to every request.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        **kwargs,
    ) -> ImagesResponse:
        """
        Generates an image asynchronously.

        Sends a POST to the `/images/generations` endpoint with the given prompt
        and model, and returns the parsed Pydantic response.

        Args:
            prompt (str): The text prompt guiding image generation.
            model (str): The model name (e.g., "dall-e-2", "stable-diffusion-v1").
            timeout (float | None): Max time (in seconds) to wait for a response.
                Defaults to 60 seconds if not provided.
            extra_headers (dict[str, str] | None): Request-specific headers
                that override any default headers.
            extra_body (object | None): Additional data to include in the
                request body, if needed.
            **kwargs: Additional generation parameters (e.g., "n", "size", "user").

        Returns:
            ImagesResponse: The parsed Pydantic model containing
                generated image URLs and metadata.
        """
        effective_timeout = 60 if timeout is None else timeout

        endpoint = f"{self.base_url}/images/generations"

        payload = {
            "model": model,
            "prompt": prompt,
            "timeout": effective_timeout,
            "extra_body": extra_body,
            **kwargs,
        }

        # Base authorization and JSON headers.
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Merge in default headers.
        headers.update(self.default_headers)
        # Merge in request-specific headers last, overwriting if a key collides
        if extra_headers:
            headers.update(extra_headers)

        client_timeout = aiohttp.ClientTimeout(total=effective_timeout)

        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                return ImagesResponse.model_validate(await resp.json())
