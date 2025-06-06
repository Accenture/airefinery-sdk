"""
Module providing clients for the embedding create endpoint.
All responses are validated using Pydantic models.

This module includes:
  - `EmbeddingsClient` for synchronous calls.
  - `AsyncEmbeddingsClient` for asynchronous calls.

All clients call the `/v1/embeddings` endpoint, and all responses
are validated using Pydantic models.

Example usage:

    sync_client = EmbeddingsClient(
        base_url="https://api.airefinery.accenture.com",
        api_key="...",
        default_headers={"X-Client-Version": "1.2.3"}  # optional base headers
    )
    response = sync_client.create(
        model="intfloat/e5-mistral-7b-instruct",
        input=["Hello world"],
        timeout=5.0,
        extra_headers={"X-Request-Id": "abc123"},      # per-request headers
        extra_body={"input_type": "query"}             # optional extra_body
    )

    async_client = AsyncEmbeddingsClient(
        base_url="https://api.airefinery.accenture.com",
        api_key="...",
        default_headers={"X-Client-Version": "1.2.3"}
    )
    response = await async_client.create(
        model="intfloat/e5-mistral-7b-instruct",
        input=["Hello async world"],
        timeout=5.0,
        extra_headers={"X-Request-Id": "xyz789"},
        extra_body={"input_type": "query"}             # optional extra_body
    )
"""

import aiohttp
import requests

from air.types import CreateEmbeddingResponse


class EmbeddingsClient:  # pylint: disable=too-few-public-methods
    """
    A synchronous client for the embedding create endpoint.

    This class handles sending requests to the embedding create endpoint
    and converts the responses into Pydantic models for type safety.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_headers: dict[str, str] | None = None,
    ):
        """
        Initializes the synchronous embeddings client.

        Args:
            base_url (str): Base URL of the API (e.g., "https://api.airefinery.accenture.com").
            api_key (str): API key for authorization.
            default_headers (dict[str, str] | None): Optional headers applied to every request.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}

    def create(
        self,
        *,
        model: str,
        input: list,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        **kwargs,
    ) -> CreateEmbeddingResponse:
        """
        Creates embeddings synchronously.

        Args:
            model (str): The model name (e.g., "intfloat/e5-mistral-7b-instruct").
            input (list): A list of texts or tokens for which embeddings should be generated.
            timeout (float | None): Maximum time (in seconds) to wait for a response
                before timing out. Defaults to 60 if not provided.
            extra_headers (dict[str, str] | None): Request-specific headers that get merged
                with any default headers for this invocation.
            extra_body (object | None): Additional data to include in the request body,
                if needed e.g., {"input_type": "query"}.
            **kwargs: Additional parameters (e.g., "user").

        Returns:
            CreateEmbeddingResponse: The parsed Pydantic model containing the embeddings.
        """

        endpoint = f"{self.base_url}/embeddings"
        payload = {"model": model, "input": input, "extra_body": extra_body, **kwargs}

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

        effective_timeout = timeout if timeout is not None else 60
        response = requests.post(
            endpoint, json=payload, headers=headers, timeout=effective_timeout
        )
        response.raise_for_status()
        return CreateEmbeddingResponse.model_validate(response.json())


class AsyncEmbeddingsClient:  # pylint: disable=too-few-public-methods
    """
    An asynchronous client for the embedding create endpoint.

    This class handles sending requests to the embedding create endpoint
    and converts the responses into Pydantic models for type safety.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_headers: dict[str, str] | None = None,
    ):
        """
        Initializes the asynchronous embeddings client.

        Args:
            base_url (str): Base URL of the API (e.g., "https://api.airefinery.accenture.com").
            api_key (str): API key for authorization.
            default_headers (dict[str, str] | None): Optional headers applied to every request.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}

    async def create(
        self,
        *,
        model: str,
        input: list,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        **kwargs,
    ) -> CreateEmbeddingResponse:
        """
        Creates embeddings asynchronously.

        Args:
            model (str): The model name (e.g., "intfloat/e5-mistral-7b-instruct").
            input (list): A list of texts or tokens for which embeddings should be generated.
            timeout (float | None): Maximum time (in seconds) to wait for a response
                before timing out. Defaults to 60 if not provided.
            extra_headers (dict[str, str] | None): Request-specific headers that get merged
                with any default headers for this invocation.
            extra_body (object | None): Additional data to include in the request body,
                if needed e.g., {"input_type": "query"}.
            **kwargs: Additional parameters (e.g., "user").

        Returns:
            CreateEmbeddingResponse: The parsed Pydantic model containing the embeddings.
        """
        endpoint = f"{self.base_url}/embeddings"
        payload = {"model": model, "input": input, "extra_body": extra_body, **kwargs}

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

        effective_timeout = 60 if timeout is None else timeout
        client_timeout = aiohttp.ClientTimeout(total=effective_timeout)

        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                return CreateEmbeddingResponse.model_validate(await resp.json())
