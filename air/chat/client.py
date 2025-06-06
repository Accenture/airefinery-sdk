"""
Module providing chat client classes (both synchronous and asynchronous)
that organize sub-clients (e.g., completions) under a `.completions` property.

This module includes:
  - `ChatClient` and `ChatCompletionsClient` for synchronous calls.
  - `AsyncChatClient` and `AsyncChatCompletionsClient` for asynchronous calls.

All clients call the `/v1/chat/completions` endpoint, and all responses
are validated using Pydantic models.

Example usage:

    from air.chat.client import ChatClient, AsyncChatClient

    # Synchronous usage:
    sync_client = ChatClient(
        base_url="https://api.airefinery.accenture.com",
        api_key="...",
        default_headers={"X-Client-Version": "1.2.3"},  # optional 'base' headers
    )
    response = sync_client.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        messages=[...],
        timeout=10.0,
        extra_headers={"X-Request-Id": "abc-123"},  # per-request additional headers
        extra_body={"input_type": "query"}          # optional extra_body
    )

    # Asynchronous usage:
    async_client = AsyncChatClient(
        base_url="https://api.airefinery.accenture.com",
        api_key="...",
        default_headers={"X-Client-Version": "1.2.3"},  # optional 'base' headers
    )
    response = await async_client.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        messages=[...],
        timeout=10.0,
        extra_headers={"X-Request-Id": "xyz-789"},  # per-request additional headers
        extra_body={"input_type": "query"}          # optional extra_body
    )
"""

import aiohttp
import requests

from air.types import ChatCompletion


class ChatCompletionsClient:  # pylint: disable=too-few-public-methods
    """
    A synchronous client for the chat completions endpoint.

    This class sends requests to create chat completions and converts
    the responses into Pydantic models for type safety.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_headers: dict[str, str] | None = None,
    ):
        """
        Initializes the synchronous completions client.

        Args:
            base_url (str): Base URL of the API (e.g. "https://api.airefinery.accenture.com").
            api_key (str): API key for authorization.
            default_headers (dict[str, str] | None): Headers to include in every request,
                e.g. {"X-Client-Version": "1.2.3"}.
        """
        self.base_url = base_url
        self.api_key = api_key
        # Store default headers, or default to an empty dict if not provided
        self.default_headers = default_headers or {}

    def create(
        self,
        *,
        model: str,
        messages: list,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        **kwargs,
    ) -> ChatCompletion:
        """
        Creates a chat completion synchronously.

        Args:
            model (str): Model name, e.g., "meta-llama/Llama-4-Maverick-17B-128E-Instruct".
            messages (list): A list of dicts with "role" and "content" for the conversation.
            timeout (float | None): Max time (in seconds) to wait before timing out.
                Defaults to 60 if not provided.
            extra_headers (dict[str, str] | None): Additional headers specific to
                this request, e.g. {"X-Correlation-Id": "..."}.
            extra_body (object | None): Additional data to include in the request body,
                if needed e.g., {"input_type": "query"}.
            **kwargs: Additional parameters like "temperature", "max_tokens", etc.

        Returns:
            ChatCompletion: The parsed completion object.
        """
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "extra_body": extra_body,
            **kwargs,
        }

        # Start with built-in auth/JSON headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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
        return ChatCompletion.model_validate(resp.json())


class ChatClient:  # pylint: disable=too-few-public-methods
    """
    A higher-level synchronous chat client that groups related API calls.

    This object provides a `.completions` property (ChatCompletionsClient),
    so you can use: client.completions.create(...)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_headers: dict[str, str] | None = None,
    ):
        """
        Initializes the synchronous chat client.

        Args:
            base_url (str): The API base URL, e.g. "https://api.airefinery.accenture.com".
            api_key (str): Your API key for authentication.
            default_headers (dict[str, str] | None): Headers that apply to all requests
                in this client, e.g., {"X-Client-Version": "1.2.3"}.
        """
        self.completions = ChatCompletionsClient(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
        )


class AsyncChatCompletionsClient:  # pylint: disable=too-few-public-methods
    """
    An asynchronous client for the chat completions endpoint.

    This class sends requests to create chat completions
    and converts the responses into Pydantic models for type safety.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_headers: dict[str, str] | None = None,
    ):
        """
        Initializes the asynchronous completions client.

        Args:
            base_url (str): Base URL of the API, e.g. "https://api.airefinery.accenture.com".
            api_key (str): API key for authorization.
            default_headers (dict[str, str] | None): Headers included in every request
                made by this client, e.g., {"X-Client-Version": "1.2.3"}.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}

    async def create(
        self,
        *,
        model: str,
        messages: list,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        **kwargs,
    ) -> ChatCompletion:
        """
        Creates a chat completion asynchronously.

        Args:
            model (str): Model name, e.g. "meta-llama/Llama-4-Maverick-17B-128E-Instruct".
            messages (list): A list of dicts with "role" and "content" for the conversation.
            timeout (float | None): Max time (in seconds) to wait before timing out.
                Defaults to 60 if not provided.
            extra_headers (dict[str, str] | None): Additional headers specific to
                this request, e.g., {"X-Correlation-Id": "..."}.
            extra_body (object | None): Additional data to include in the request body,
                if needed e.g., {"input_type": "query"}.
            **kwargs: Additional parameters like "temperature", "max_tokens", etc.

        Returns:
            ChatCompletion: The parsed completion object.
        """
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "extra_body": extra_body,
            **kwargs,
        }

        # Start with built-in auth/JSON headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Merge in default_headers
        headers.update(self.default_headers)
        # Merge in extra_headers
        if extra_headers:
            headers.update(extra_headers)

        client_timeout = aiohttp.ClientTimeout(total=timeout if timeout else 60)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                return ChatCompletion.model_validate(await resp.json())


class AsyncChatClient:  # pylint: disable=too-few-public-methods
    """
    A higher-level asynchronous chat client that groups related API calls.

    This object provides a `.completions` property (AsyncChatCompletionsClient),
    so you can use: await client.completions.create(...)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_headers: dict[str, str] | None = None,
    ):
        """
        Initializes the asynchronous chat client.

        Args:
            base_url (str): The API base URL, e.g. "https://api.airefinery.accenture.com".
            api_key (str): Your API key for authentication.
            default_headers (dict[str, str] | None): Headers that apply to all requests
                in this client, e.g. {"X-Client-Version": "1.2.3"}.
        """
        self.completions = AsyncChatCompletionsClient(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
        )
