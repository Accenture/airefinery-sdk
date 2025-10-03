"""
Module providing clients for the moderation create endpoint.
All responses are validated using Pydantic models.

This module includes:
  - `ModerationsClient` for synchronous calls.
  - `AsyncModerationsClient` for asynchronous calls.

All clients call the `/v1/moderations` endpoint, and all responses
are validated using Pydantic models.

Example usage:

    sync_client = ModerationsClient(
        base_url="https://api.airefinery.accenture.com",
        api_key="...",
    )
    response = sync_client.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        input=["Hello world"],
        timeout=5.0,
        extra_headers={"X-Request-Id": "abc123"},      # per-request headers
        extra_body={"input_type": "query"}             # optional extra_body
    )

    async_client = AsyncModerationsClient(
        base_url="https://api.airefinery.accenture.com",
        api_key="...",
    )
    response = await async_client.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        input=["Hello async world"],
        timeout=5.0,
        extra_headers={"X-Request-Id": "xyz789"},
        extra_body={"input_type": "query"}             # optional extra_body
    )
"""

from typing import Iterable, Sequence, Union

import aiohttp
import requests
from openai.types.moderation_multi_modal_input_param import (
    ModerationMultiModalInputParam,
)

from air import BASE_URL, __version__
from air.auth.token_provider import TokenProvider
from air.types import ModerationCreateResponse
from air.utils import get_base_headers

ENDPOINT_MODERATIONS = "{base_url}/v1/moderations"


class ModerationsClient:  # pylint: disable=too-few-public-methods
    """
    A synchronous client for the moderation create endpoint.

    This class handles sending requests to the moderation create endpoint
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
        Initializes the synchronous moderations client.

        Args:
            api_key (str): API key for authorization.
            base_url (str): Base URL of the API (e.g., "https://api.airefinery.accenture.com").
            default_headers (dict[str, str] | None): Optional headers applied to every request.
        """
        self.base_url = base_url
        self.api_key = api_key

        self.default_headers = default_headers or {}

    def create(
        self,
        *,
        model: str,
        input: Union[str, Sequence[str], Iterable[ModerationMultiModalInputParam]],
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        **kwargs,
    ) -> ModerationCreateResponse:
        """
        Creates moderations synchronously.

        Args:
            model (str): The model name (e.g., "meta-llama/Llama-4-Maverick-17B-128E-Instruct").
            input (list): A list of texts or tokens for which moderations should be generated.
            timeout (float | None): Maximum time (in seconds) to wait for a response
                before timing out. Defaults to 60 if not provided.
            extra_headers (dict[str, str] | None): Request-specific headers that get merged
                with any default headers for this invocation.
            extra_body (object | None): Additional data to include in the request body,
                if needed e.g., {"input_type": "query"}.
            **kwargs: Additional parameters (e.g., "user").

        Returns:
            CreatemoderationResponse: The parsed Pydantic model containing the moderations.
        """

        endpoint = ENDPOINT_MODERATIONS.format(base_url=self.base_url)

        payload = {"model": model, "input": input, "extra_body": extra_body, **kwargs}

        # Base authorization and JSON headers.
        headers = get_base_headers(self.api_key)

        # Merge in default headers
        headers.update(self.default_headers)
        # Merge in request-specific headers last, overwriting if a key collides
        if extra_headers:
            headers.update(extra_headers)

        effective_timeout = timeout if timeout is not None else 60
        response = requests.post(
            endpoint, json=payload, headers=headers, timeout=effective_timeout
        )
        response.raise_for_status()
        return ModerationCreateResponse.model_validate(response.json())


class AsyncModerationsClient:  # pylint: disable=too-few-public-methods
    """
    An asynchronous client for the moderation create endpoint.

    This class handles sending requests to the moderation create endpoint
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
        Initializes the asynchronous moderations client.

        Args:
            api_key (str): API key for authorization.
            base_url (str): Base URL of the API (e.g., "https://api.airefinery.accenture.com").
            default_headers (dict[str, str] | None): Optional headers applied to every request.
        """
        self.base_url = base_url

        self.api_key = api_key
        self.default_headers = default_headers or {}

    async def create(
        self,
        *,
        model: str,
        input: Union[str, Sequence[str], Iterable[ModerationMultiModalInputParam]],
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        **kwargs,
    ) -> ModerationCreateResponse:
        """
        Creates moderations asynchronously.

        Args:
            model (str): The model name (e.g., "intfloat/e5-mistral-7b-instruct").
            input (list): A list of texts or tokens for which moderations should be generated.
            timeout (float | None): Maximum time (in seconds) to wait for a response
                before timing out. Defaults to 60 if not provided.
            extra_headers (dict[str, str] | None): Request-specific headers that get merged
                with any default headers for this invocation.
            extra_body (object | None): Additional data to include in the request body,
                if needed e.g., {"input_type": "query"}.
            **kwargs: Additional parameters (e.g., "user").

        Returns:
            CreatemoderationResponse: The parsed Pydantic model containing the moderations.
        """
        endpoint = ENDPOINT_MODERATIONS.format(base_url=self.base_url)

        headers = get_base_headers(self.api_key)

        payload = {"model": model, "input": input, "extra_body": extra_body, **kwargs}

        # Merge in default headers.
        headers.update(self.default_headers)
        # Merge in request-specific headers last, overwriting if a key collides
        if extra_headers:
            headers.update(extra_headers)

        effective_timeout = 60 if timeout is None else timeout
        client_timeout = aiohttp.ClientTimeout(total=effective_timeout)

        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                return ModerationCreateResponse.model_validate(await resp.json())
