"""
Module providing a client for fine-tuning operations.
All responses are validated using Pydantic models.

This module includes:
  - `FineTuningClient` for synchronous calls.
  - `AsyncFineTuningClient` for asynchronous calls.
  - A `.jobs` namespace with methods:
      * create
      * cancel
      * list_events
"""

from typing import Any, Dict, List, Optional

import aiohttp
import requests

from air import BASE_URL, __version__
from air.auth import TokenProvider
from air.types import FineTuningRequest
from air.types.constants import DEFAULT_TIMEOUT
from air.types.fine_tuning import FineTuningJobConfig
from air.utils import get_base_headers, get_base_headers_async

ENDPOINT_TAAS_LAUNCH = "{base_url}/v1/fine_tuning/jobs"
ENDPOINT_TAAS_CANCEL = "{base_url}/v1/fine_tuning/jobs/{job_id}/cancel"
ENDPOINT_TAAS_EVENTS = "{base_url}/v1/fine_tuning/jobs/{job_id}/events"


class FineTuningClient:
    """
    A synchronous client for fine-tuning related endpoints.

    This class handles sending requests to fine-tuning endpoints
    and converts the responses into Pydantic models for type safety.
    """

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str = BASE_URL,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Initializes the synchronous fine-tuning client.

        Args:
            api_key (str | TokenProvider): Credential that will be placed in the
                ``Authorization`` header of every request.
                * **str** – a raw bearer token / API key.
                * **TokenProvider** – an object whose ``token()`` (and
                  ``token_async()``) method returns a valid bearer token.  The
                  client calls the provider automatically before each request.
            base_url (str, optional): Base URL of the API
                Defaults to BASE_URL.
            default_headers (dict[str, str] | None):
                Optional headers applied to every request
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_headers = default_headers.copy() if default_headers else {}

    @property
    def jobs(self) -> "FineTuningClient._Jobs":
        """
        Returns the jobs sub-client for managing fine-tuning jobs.
        """
        return self._Jobs(self)

    class _Jobs:
        def __init__(self, parent: "FineTuningClient"):
            self._parent = parent

        def _headers(self, extra_headers: Optional[Dict[str, str]]) -> Dict[str, str]:
            headers = get_base_headers(self._parent.api_key, extra_headers)
            headers.update(self._parent.default_headers)
            return headers

        def create(
            self,
            *,
            job_config: dict[str, Any] | FineTuningJobConfig,
            uuid: str,
            timeout: float | None = None,
            extra_headers: Optional[Dict[str, str]] = None,
            **kwargs,
        ) -> FineTuningRequest:
            """
            Create a new fine-tuning job.

            Args:
                job_config (dict[str, Any]): Job config for fine-tuning request
                uuid (str): A unique identifier for the user
                timeout (float | None): Max seconds to wait (defaults to DEFAULT_TIMEOUT)
                extra_headers (dict | None): Request-specific headers
                **kwargs: Additional generation parameters

            Returns:
                FineTuningRequest: The created job object.
            """
            # Validate job_config is valid following FineTuningJobConfig
            if not isinstance(job_config, FineTuningJobConfig):
                try:
                    job_config = FineTuningJobConfig.model_validate(job_config)
                except Exception as e:
                    raise ValueError(f"Invalid FineTuningJobConfig: {e}") from e

            effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
            endpoint = ENDPOINT_TAAS_LAUNCH.format(base_url=self._parent.base_url)

            payload: Dict[str, Any] = {
                "config": job_config.model_dump(),
                "uuid": uuid,
                **kwargs,
            }

            headers = self._headers(extra_headers)

            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=effective_timeout,
            )
            response.raise_for_status()
            return FineTuningRequest.model_validate(response.json())

        def cancel(
            self,
            *,
            uuid: str,
            fine_tuning_job_id: str,
            timeout: float | None = None,
            extra_headers: Optional[Dict[str, str]] = None,
        ) -> FineTuningRequest:
            """
            Cancel a fine-tuning job.

            Args:
                uuid (str): A unique identifier for the user
                fine_tuning_job_id (str): ID of the fine-tuning job to cancel.
                timeout (float | None): Max seconds to wait (defaults to DEFAULT_TIMEOUT).
                extra_headers (dict | None): Request-specific headers.

            Returns:
                FineTuningRequest: The cancellation confirmation.
            """
            effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
            endpoint = ENDPOINT_TAAS_CANCEL.format(
                base_url=self._parent.base_url, job_id=fine_tuning_job_id
            )

            payload = {
                "uuid": uuid,
                "fine_tuning_job_id": fine_tuning_job_id,
            }

            headers = self._headers(extra_headers)

            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=effective_timeout,
            )
            response.raise_for_status()
            return FineTuningRequest.model_validate(response.json())

        def list_events(
            self,
            *,
            uuid: str,
            fine_tuning_job_id: str,
            timeout: float | None = None,
            extra_headers: Optional[Dict[str, str]] = None,
        ) -> List[Dict[str, Any]]:
            """
            Retrieve status update events for a specific fine-tuning job.

            This method fetches the events generated during the lifecycle of a fine-tuning job,
            allowing you to monitor the job's progress, track fine-tuning steps, and check for any
            warnings or errors.

            Args:
                uuid (str): A unique identifier for the user
                fine_tuning_job_id (str):
                    The unique identifier of the fine-tuning job for which to retrieve events.
                timeout (float | None): Max seconds to wait (defaults to DEFAULT_TIMEOUT).
                extra_headers (dict | None): Request-specific headers.


            Returns:
                List[Dict[str, Any]]:
                    An object containing a list of fine-tuning event objects, possibly along with
                    pagination information for fetching additional events.
            """
            effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
            endpoint = ENDPOINT_TAAS_EVENTS.format(
                base_url=self._parent.base_url, job_id=fine_tuning_job_id
            )

            # Query payload
            payload = {"uuid": uuid, "fine_tuning_job_id": fine_tuning_job_id}
            # Prepare headers (including auth)
            headers = self._headers(extra_headers)
            # Send GET request to retrieve events
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=effective_timeout,
            )
            response.raise_for_status()
            # Validate and return the response model
            return response.json()


class AsyncFineTuningClient:
    """
    An asynchronous client for fine-tuning related endpoints.

    This class handles sending requests to fine-tuning endpoints
    and converts the responses into Pydantic models for type safety.
    """

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str = BASE_URL,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Initializes the asynchronous fine-tuning client.

        Args:
            api_key (str | TokenProvider): Credential that will be placed in the
                ``Authorization`` header of every request.
                * **str** – a raw bearer token / API key.
                * **TokenProvider** – an object whose ``token()`` (and
                  ``token_async()``) method returns a valid bearer token.  The
                  client calls the provider automatically before each request.
            base_url (str, optional): Base URL of the API
                Defaults to BASE_URL.
            default_headers (dict[str, str] | None):
                Optional headers applied to every request
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_headers = default_headers.copy() if default_headers else {}

    @property
    def jobs(self) -> "AsyncFineTuningClient._Jobs":
        """
        Returns the jobs sub-client for managing fine-tuning jobs.
        """
        return self._Jobs(self)

    class _Jobs:
        def __init__(self, parent: "AsyncFineTuningClient"):
            self._parent = parent

        async def _headers(
            self, extra_headers: Optional[Dict[str, str]]
        ) -> Dict[str, str]:
            headers = await get_base_headers_async(self._parent.api_key, extra_headers)
            headers.update(self._parent.default_headers)
            return headers

        async def create(
            self,
            *,
            job_config: dict[str, Any] | FineTuningJobConfig,
            uuid: str,
            timeout: float | None = None,
            extra_headers: Optional[Dict[str, str]] = None,
            **kwargs,
        ) -> FineTuningRequest:
            """
            Create a new fine-tuning job asynchronously.

            Args:
                job_config (dict[str, Any]): Job config for fine-tuning request
                uuid (str): A unique identifier for the user
                timeout (float | None): Max seconds to wait (defaults to DEFAULT_TIMEOUT)
                extra_headers (dict | None): Request-specific headers
                **kwargs: Additional generation parameters

            Returns:
                FineTuningRequest: The created job object.
            """
            # Validate job_config is valid following FineTuningJobConfig
            if not isinstance(job_config, FineTuningJobConfig):
                try:
                    job_config = FineTuningJobConfig.model_validate(job_config)
                except Exception as e:
                    raise ValueError(f"Invalid FineTuningJobConfig: {e}") from e

            effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
            endpoint = ENDPOINT_TAAS_LAUNCH.format(base_url=self._parent.base_url)

            payload: Dict[str, Any] = {
                "config": job_config.model_dump(),
                "uuid": uuid,
                **kwargs,
            }

            headers = await self._headers(extra_headers)

            client_timeout = aiohttp.ClientTimeout(total=effective_timeout)

            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(
                    endpoint, json=payload, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    return FineTuningRequest.model_validate(await resp.json())

        async def cancel(
            self,
            *,
            uuid: str,
            fine_tuning_job_id: str,
            timeout: float | None = None,
            extra_headers: Optional[Dict[str, str]] = None,
        ) -> FineTuningRequest:
            """
            Cancel a fine-tuning job asynchronously.

            Args:
                uuid (str): A unique identifier for the user
                fine_tuning_job_id (str): ID of the fine-tuning job to cancel.
                timeout (float | None): Max seconds to wait (defaults to DEFAULT_TIMEOUT).
                extra_headers (dict | None): Request-specific headers.

            Returns:
                FineTuningRequest: The cancellation confirmation.
            """
            effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
            endpoint = ENDPOINT_TAAS_CANCEL.format(
                base_url=self._parent.base_url, job_id=fine_tuning_job_id
            )

            payload = {
                "uuid": uuid,
                "fine_tuning_job_id": fine_tuning_job_id,
            }

            headers = await self._headers(extra_headers)

            client_timeout = aiohttp.ClientTimeout(total=effective_timeout)

            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(
                    endpoint, json=payload, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    return FineTuningRequest.model_validate(await resp.json())

        async def list_events(
            self,
            *,
            uuid: str,
            fine_tuning_job_id: str,
            timeout: float | None = None,
            extra_headers: Optional[Dict[str, str]] = None,
        ) -> List[Dict[str, Any]]:
            """
            Retrieve status update events for a specific fine-tuning job asynchronously.

            This method fetches the events generated during the lifecycle of a fine-tuning job,
            allowing you to monitor the job's progress, track fine-tuning steps, and check for any
            warnings or errors.

            Args:
                uuid (str): A unique identifier for the user.
                fine_tuning_job_id (str):
                    The unique identifier of the fine-tuning job for which to retrieve events.
                timeout (float | None): Max seconds to wait (defaults to DEFAULT_TIMEOUT).
                extra_headers (dict | None): Request-specific headers.

            Returns:
                List[Dict[str, Any]]:
                    A list of fine-tuning event objects, possibly along with
                    pagination information for fetching additional events.
            """
            effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
            endpoint = ENDPOINT_TAAS_EVENTS.format(
                base_url=self._parent.base_url, job_id=fine_tuning_job_id
            )

            # Query payload
            payload = {"uuid": uuid, "fine_tuning_job_id": fine_tuning_job_id}
            # Prepare headers (including auth)
            headers = await self._headers(extra_headers)

            client_timeout = aiohttp.ClientTimeout(total=effective_timeout)

            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(
                    endpoint, json=payload, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
