"""SimFarm SDK client for managing simulation jobs.

This module provides synchronous and asynchronous clients for interacting with
the SimFarm service API. The clients support job submission, monitoring,
cancellation, and log retrieval operations for simulation workloads.

Classes:
    SimFarmClient: Synchronous client for SimFarm operations.
    AsyncSimFarmClient: Asynchronous client for SimFarm operations.

Both clients provide methods for:
    - Getting detailed job information
    - Listing jobs with filtering options
"""

from __future__ import annotations

import aiohttp
import requests
from typing import Optional

from air import BASE_URL
from air.auth.token_provider import TokenProvider
from air.types.constants import DEFAULT_TIMEOUT
from air.types.simfarm import SimfarmJobDetailsResponse, SimfarmJobListResponse
from air.utils.get_base_headers import get_base_headers, get_base_headers_async

# URL Templates
SIMFARM_JOB_DETAILS_URL = "{base_url}/simfarm/jobs/{job_id}"
SIMFARM_LIST_JOBS_URL = "{base_url}/simfarm/list_jobs"


def _effective_timeout(timeout: float | None) -> float:
    """Get the effective timeout value.

    Args:
        timeout: Timeout in seconds, or None to use default.

    Returns:
        float: The effective timeout value.
    """
    return timeout if timeout is not None else DEFAULT_TIMEOUT


class SimFarmClient:
    """Synchronous client for SimFarm SQL endpoints."""

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str = BASE_URL,
        default_headers: dict[str, str] | None = None,
    ):
        """Initialize the SimFarm client.

        Args:
            api_key: API key for authentication with the SimFarm service.
            base_url: Base URL for the SimFarm API. Defaults to the global AIRefinery endpoint.
            default_headers: Additional headers to include in all requests.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_headers = default_headers or {}

    def get_job_details(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> SimfarmJobDetailsResponse:
        """Get detailed information about a specific job.

        Args:
            job_id: The unique identifier of the job.
            timeout: Request timeout in seconds. Defaults to DEFAULT_TIMEOUT.
            extra_headers: Additional headers to include in the request.

        Returns:
            SimfarmJobDetailsResponse: Detailed information about the requested job.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        headers = get_base_headers(
            self.api_key, {**self.default_headers, **(extra_headers or {})}
        )
        url = SIMFARM_JOB_DETAILS_URL.format(base_url=self.base_url, job_id=job_id)

        response = requests.get(
            url,
            headers=headers,
            timeout=_effective_timeout(timeout),
        )
        response.raise_for_status()
        return SimfarmJobDetailsResponse.model_validate(response.json())

    def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> SimfarmJobListResponse:
        """List jobs with optional filtering.

        Args:
            status: Filter jobs by status (e.g., 'running', 'completed', 'failed').
            limit: Maximum number of jobs to return.
            timeout: Request timeout in seconds. Defaults to DEFAULT_TIMEOUT.
            extra_headers: Additional headers to include in the request.

        Returns:
            SimfarmJobListResponse: List of jobs matching the filter criteria.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        headers = get_base_headers(
            self.api_key, {**self.default_headers, **(extra_headers or {})}
        )
        params = {}
        if status:
            params["status"] = status
        if limit:
            params["limit"] = limit

        url = SIMFARM_LIST_JOBS_URL.format(base_url=self.base_url)
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=_effective_timeout(timeout),
        )
        response.raise_for_status()
        return SimfarmJobListResponse.model_validate(response.json())


class AsyncSimFarmClient:
    """Asynchronous client for SimFarm SQL endpoints."""

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str = BASE_URL,
        default_headers: dict[str, str] | None = None,
    ):
        """Initialize the async SimFarm client.

        Args:
            api_key: API key for authentication with the SimFarm service.
            base_url: Base URL for the SimFarm API. Defaults to the global AIRefinery endpoint.
            default_headers: Additional headers to include in all requests.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_headers = default_headers or {}

    async def get_job_details(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> SimfarmJobDetailsResponse:
        """Get detailed information about a specific job.

        Args:
            job_id: The unique identifier of the job.
            timeout: Request timeout in seconds. Defaults to DEFAULT_TIMEOUT.
            extra_headers: Additional headers to include in the request.

        Returns:
            SimfarmJobDetailsResponse: Detailed information about the requested job.

        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        headers = await get_base_headers_async(
            self.api_key, {**self.default_headers, **(extra_headers or {})}
        )
        url = SIMFARM_JOB_DETAILS_URL.format(base_url=self.base_url, job_id=job_id)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=_effective_timeout(timeout)),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return SimfarmJobDetailsResponse.model_validate(data)

    async def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> SimfarmJobListResponse:
        """List jobs with optional filtering.

        Args:
            status: Filter jobs by status (e.g., 'running', 'completed', 'failed').
            limit: Maximum number of jobs to return.
            timeout: Request timeout in seconds. Defaults to DEFAULT_TIMEOUT.
            extra_headers: Additional headers to include in the request.

        Returns:
            SimfarmJobListResponse: List of jobs matching the filter criteria.

        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        headers = get_base_headers(
            self.api_key, {**self.default_headers, **(extra_headers or {})}
        )
        params = {}
        if status:
            params["status"] = status
        if limit:
            params["limit"] = limit

        url = SIMFARM_LIST_JOBS_URL.format(base_url=self.base_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=_effective_timeout(timeout)),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return SimfarmJobListResponse.model_validate(data)
