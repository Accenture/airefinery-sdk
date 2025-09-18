"""
Module providing clients for image operations.
All responses are validated using Pydantic models.

This module includes:
  - `ImagesClient` for synchronous calls.
  - `AsyncImagesClient` for asynchronous calls.

Both clients call eitehr the `/images/generations` endpoint, or the
`/images/segmentations` endpoint.
All responses are validated using Pydantic models (`ImagesResponse`,
`SegmentationResponse`).
"""

import aiohttp
import requests

from air import BASE_URL, __version__
from air.auth.token_provider import TokenProvider
from air.types import ImagesResponse, SegmentationResponse
from air.types.constants import DEFAULT_TIMEOUT
from air.utils import get_base_headers, get_base_headers_async

ENDPOINT_IMAGE_GENERATIONS = "{base_url}/v1/images/generations"
ENDPOINT_IMAGE_SEGMENTATIONS = "{base_url}/v1/images/segmentations"


class ImagesClient:
    """
    A synchronous client for image related endpoints.

    This class handles sending requests to image related endpoints
    and converts the responses into Pydantic models for type safety.
    """

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str = BASE_URL,
        default_headers: dict[str, str] | None = None,
    ):
        """
        Initializes the synchronous image client.

        Args:
            api_key (str | TokenProvider): Credential that will be placed in the
                ``Authorization`` header of every request.
                * **str** – a raw bearer token / API key.
                * **TokenProvider** – an object whose ``token()`` (and
                  ``token_async()``) method returns a valid bearer token.  The
                  client calls the provider automatically before each request.

            base_url (str): Base URL of the API (e.g., "https://api.airefinery.accenture.com")
            default_headers (dict[str, str] | None): Optional headers applied to every request
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
            prompt (str): The text prompt guiding image generation
            model (str): The model name (e.g., "black-forest-labs/FLUX.1-schnell")
            timeout (float | None): Max time (in seconds) to wait for a response.
                Defaults to 60 seconds if not provided
            extra_headers (dict[str, str] | None): Request-specific headers
                that override any default headers
            extra_body (object | None): Additional data to include in the
                request body, if needed
            **kwargs: Additional generation parameters (e.g., "n", "size", "user")

        Returns:
            ImagesResponse: The parsed Pydantic model containing
                generated image URLs and metadata
        """
        effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

        endpoint = ENDPOINT_IMAGE_GENERATIONS.format(base_url=self.base_url)

        payload = {
            "model": model,
            "prompt": prompt,
            "timeout": effective_timeout,
            "extra_body": extra_body,
            **kwargs,
        }

        # Base authorization and JSON headers.
        headers = get_base_headers(self.api_key)

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

    def segment(
        self,
        *,
        image: str,
        segment_prompt: list[list[list[int]]],
        model: str,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        **kwargs,
    ) -> SegmentationResponse:
        """
        Performs point-prompt-based segmentation synchronously

        Args:
            image (str): bse64-encoded image.
            segment_prompt (list[list[list[int]]]): Nested list of points for segmentation, formatted as [[[x, y], ...]]
            model (str): The segmentation model (e.g., "syscv-community/sam-hq-vit-base")
            timeout (float | None): Max time (in seconds) to wait for a response.
                Defaults to 60 seconds if not provided
            extra_headers (dict[str, str] | None): Request-specific headers
                that override any default headers
            extra_body (object | None): Additional data to include in the
                request body, if needed
            **kwargs: Additional generation parameters (e.g., "n", "size", "user")

        Returns:
            SegmentationResponse: The parsed Pydantic model containing a list of
                base64-encoded categorical mask images
        """
        effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

        endpoint = ENDPOINT_IMAGE_SEGMENTATIONS.format(base_url=self.base_url)

        payload = {
            "model": model,
            "image_prompt": image,
            "segment_prompt": segment_prompt,
            "extra_body": extra_body,
            **kwargs,
        }

        headers = get_base_headers(self.api_key)

        headers.update(self.default_headers)
        if extra_headers:
            headers.update(extra_headers)

        response = requests.post(
            endpoint, json=payload, headers=headers, timeout=effective_timeout
        )
        response.raise_for_status()
        return SegmentationResponse.model_validate(response.json())


class AsyncImagesClient:
    """
    An asynchronous client for the image endpoint.

    This class handles sending requests to the image endpoint
    and converts the responses into Pydantic models for type safety.
    """

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str,
        default_headers: dict[str, str] | None = None,
    ):
        """
        Initializes the asynchronous image client.

        Args:
            api_key (str | TokenProvider): Credential that will be placed in the
                ``Authorization`` header of every request.
                * **str** – a raw bearer token / API key.
                * **TokenProvider** – an object whose ``token()`` (and
                  ``token_async()``) method returns a valid bearer token.  The
                  client calls the provider automatically before each request.

            base_url (str): Base URL of the API (e.g., "https://api.airefinery.accenture.com")
            default_headers (dict[str, str] | None): Optional headers applied to every request
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
            model (str): The model name (e.g., "black-forest-labs/FLUX.1-schnell")
            timeout (float | None): Max time (in seconds) to wait for a response.
                Defaults to 60 seconds if not provided
            extra_headers (dict[str, str] | None): Request-specific headers
                that override any default headers
            extra_body (object | None): Additional data to include in the
                request body, if needed
            **kwargs: Additional generation parameters

        Returns:
            ImagesResponse: The parsed Pydantic model containing
                generated image URLs and metadata
        """
        effective_timeout = DEFAULT_TIMEOUT if timeout is None else timeout

        endpoint = ENDPOINT_IMAGE_GENERATIONS.format(base_url=self.base_url)

        payload = {
            "model": model,
            "prompt": prompt,
            "timeout": effective_timeout,
            "extra_body": extra_body,
            **kwargs,
        }

        # Base authorization and JSON headers.
        headers = await get_base_headers_async(self.api_key)

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

    async def segment(
        self,
        *,
        image: str,
        segment_prompt: list[list[list[int]]],
        model: str,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        **kwargs,
    ) -> SegmentationResponse:
        """
        Performs point-prompt-based segmentation asynchronously

        Args:
            image (str): Base64-encoded image.
            segment_prompt (list[list[list[int]]]): Nested list of points for segmentation, formatted as [[[x, y], ...]]
            model (str): The segmentation model (e.g., "syscv-community/sam-hq-vit-base")
            timeout (float | None): Max time (in seconds) to wait for a response.
                Defaults to 60 seconds if not provided
            extra_headers (dict[str, str] | None): Request-specific headers
                that override any default headers
            extra_body (object | None): Additional data to include in the
                request body, if needed
            **kwargs: Additional generation parameters

        Returns:
            SegmentationResponse: The parsed Pydantic model containing
                generated image segments and metadata
        """
        effective_timeout = DEFAULT_TIMEOUT if timeout is None else timeout

        endpoint = ENDPOINT_IMAGE_SEGMENTATIONS.format(base_url=self.base_url)

        payload = {
            "model": model,
            "image_prompt": image,
            "segment_prompt": segment_prompt,
            "extra_body": extra_body,
            **kwargs,
        }

        headers = await get_base_headers_async(self.api_key)

        headers.update(self.default_headers)
        if extra_headers:
            headers.update(extra_headers)

        client_timeout = aiohttp.ClientTimeout(total=effective_timeout)

        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                return SegmentationResponse.model_validate(await resp.json())
