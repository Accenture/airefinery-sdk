import logging
from typing import Any, Dict

import httpx

from air.auth.token_provider import TokenProvider
from air.utils import get_base_headers, get_base_headers_async

logger = logging.getLogger(__name__)


class ObservabilityClient:
    """
    Base client for querying AI Refinery observability services
    (metrics, traces, logs).

    The server-side authentication middleware automatically resolves the
    ``organization_id`` from the bearer token, so the client does not need
    to pre-fetch it.

    Parameters:
        base_url: The base URL of the AIRefinery service.
        api_key: The API key or :class:`TokenProvider` for authentication.
        timeout: The timeout for HTTP requests in seconds.
        endpoint: The endpoint path for the observability service.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | TokenProvider | None = None,
        timeout: int = 15,
        endpoint: str = "",
    ):
        self.base_url = base_url
        self.api_key = api_key if api_key is not None else ""
        self.timeout = timeout
        self.endpoint = endpoint

    async def query(self, **parameters) -> Dict[str, Any]:
        """Query data from the observability service.

        Keyword arguments are serialised as the JSON request body and sent
        to the configured endpoint.  The server resolves the caller's
        organisation from the bearer token automatically.

        Returns:
            The parsed JSON response from the server.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx status.
        """
        headers = await get_base_headers_async(self.api_key)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}{self.endpoint}",
                json=parameters,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()


class TracesClient(ObservabilityClient):
    """Client for querying traces from the observability service."""

    def __init__(
        self,
        base_url: str,
        api_key: str | TokenProvider | None = None,
        timeout: int = 15,
    ):
        super().__init__(base_url, api_key, timeout, endpoint="/observability/traces")


class LogsClient(ObservabilityClient):
    """Client for querying logs from the observability service."""

    def __init__(
        self,
        base_url: str,
        api_key: str | TokenProvider | None = None,
        timeout: int = 15,
    ):
        super().__init__(base_url, api_key, timeout, endpoint="/observability/logs")


class MetricsClient(ObservabilityClient):
    """Client for querying metrics from the observability service."""

    def __init__(
        self,
        base_url: str,
        api_key: str | TokenProvider | None = None,
        timeout: int = 15,
    ):
        super().__init__(base_url, api_key, timeout, endpoint="/observability/metrics")
