"""Module containing the CBInsightsExecutor for CB Insights API integration."""

import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict

import httpx

from air.distiller.executor.executor import Executor
from air.types.distiller.executor.cb_insights_config import CBInsightsAgentConfig

logger = logging.getLogger(__name__)


class CBInsightsExecutor(Executor):
    """
    Executor class for the CB Insights Agent.
    """

    agent_class: str = "CBInsightsAgent"

    def __init__(
        self,
        func: Dict[str, Callable],
        send_queue: asyncio.Queue,
        account: str,
        project: str,
        uuid: str,
        role: str,
        utility_config: Dict[str, Any],
        return_string: bool = True,
    ):
        """
        Initializes the CBInsightsExecutor.
        """
        logger.debug(
            "Initializing CBInsightsExecutor with role=%r, account=%r, "
            "project=%r, uuid=%r",
            role,
            account,
            project,
            uuid,
        )

        # Casting utility config to class-specific pydantic BaseModel
        cbinsights_agent_config = CBInsightsAgentConfig(**utility_config)

        # Retrieve required fields for authentication
        client_id = os.getenv(cbinsights_agent_config.client_id, "")
        if not client_id:
            logger.error(
                "%s is not set in the environment.", cbinsights_agent_config.client_id
            )
            raise RuntimeError("client_id is not set in the environment.")

        client_secret = os.getenv(cbinsights_agent_config.client_secret, "")
        if not client_secret:
            logger.error(
                "%s is not set in the environment.",
                cbinsights_agent_config.client_secret,
            )
            raise RuntimeError("client_secret is not set in the environment.")

        # Store configuration
        self._api_base_url = cbinsights_agent_config.api_base_url
        self._wait_time = cbinsights_agent_config.wait_time

        # Initialize chat session ID (will be set from API response)
        self._chat_id = None

        # Initialize persistent httpx client as None
        self._httpx_client = None

        # Get authentication token during initialization
        try:
            self._auth_token = self._get_auth_token(
                client_id, client_secret, self._api_base_url
            )
            logger.info("Successfully retrieved CB Insights authentication token.")
        except Exception as e:
            logger.exception("Failed to retrieve CB Insights authentication token")
            raise RuntimeError(
                "Failed to retrieve CB Insights authentication token"
            ) from e

        # Initialize the base Executor with our specialized execution method.
        super().__init__(
            func=self._execute_agent,
            send_queue=send_queue,
            account=account,
            project=project,
            uuid=uuid,
            role=role,
            return_string=return_string,
        )

    async def _get_httpx_client(self) -> httpx.AsyncClient:
        """
        Get or create the persistent httpx client.

        Creates client once and reuses it for subsequent requests.
        """
        if self._httpx_client is None:
            logger.debug("Creating new persistent httpx client for CB Insights")
            self._httpx_client = httpx.AsyncClient(timeout=self._wait_time)
        else:
            logger.debug("Reusing existing httpx client for CB Insights")

        return self._httpx_client

    def _get_auth_token(
        self, client_id: str, client_secret: str, api_base_url: str
    ) -> str:
        """
        Get authentication token from CB Insights API.
        """
        auth_payload = {"clientId": client_id, "clientSecret": client_secret}
        auth_headers = {"Content-Type": "application/json"}

        try:
            with httpx.Client(timeout=self._wait_time) as client:
                response = client.post(
                    f"{api_base_url}/v2/authorize",
                    headers=auth_headers,
                    json=auth_payload,
                )
                response.raise_for_status()

                token = response.json().get("token")
                if not token:
                    raise RuntimeError("Token not found in response.")
                return token

        except httpx.HTTPError as e:
            logger.exception(
                "Failed to fetch authentication token from CB Insights API"
            )
            raise RuntimeError(
                "Failed to fetch authentication token from CB Insights API."
            ) from e
        except Exception as e:
            logger.exception("Failed to authenticate with CB Insights API")
            raise RuntimeError("Failed to authenticate with CB Insights API.") from e

    async def _execute_agent(self, **kwargs) -> str:
        """
        Executes the CB Insights Agent using a prompt.
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                "Missing 'prompt' parameter in CBInsightsExecutor._execute_agent."
            )

        logger.debug("Running CB Insights Agent with prompt=%r", prompt)
        return await self.call_chatcbi_api(prompt)

    async def call_chatcbi_api(self, prompt: str) -> str:
        """
        Send a chat query to CB Insights API and return the response.
        """
        response = await self._send_chat_request(prompt)
        return self._parse_chat_response(response)

    async def _send_chat_request(self, prompt: str) -> httpx.Response:
        """
        Send HTTP request to CB Insights ChatCBI API.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._auth_token}",
        }
        payload = {"message": prompt}
        if self._chat_id:
            payload["chatID"] = self._chat_id

        # Use persistent client
        client = await self._get_httpx_client()

        try:
            logger.debug("Sending chat request to CB Insights API (persistent client)")
            response = await client.post(
                f"{self._api_base_url}/v2/chatcbi",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return response

        except httpx.HTTPError as e:
            logger.exception("Failed to get response from CB Insights API: %s", e)
            raise RuntimeError("Failed to get response from CB Insights API") from e
        except Exception as e:
            logger.exception("Unexpected error while querying CB Insights: %s", e)
            raise RuntimeError("Unexpected error while querying CB Insights") from e

    def _parse_chat_response(self, response: httpx.Response) -> str:
        """
        Parse CB Insights API response and extract message with sources.
        """
        try:
            data = response.json()
            message = data.get("message", "")

            # Extract and store chatID for future requests
            if "chatID" in data:
                self._chat_id = data["chatID"]
                logger.debug("Updated chat session ID: %s", self._chat_id)

            if not message:
                logger.warning("Empty message received from CB Insights API")
                return "No response received from CB Insights."

            # Extract and append sources if available
            sources = data.get("sources", [])
            if sources:
                message += self._format_sources(sources)

            logger.info("CB Insights response received (len=%d)", len(message))
            return message

        except json.JSONDecodeError as e:
            logger.exception("Failed to parse CB Insights API response as JSON")
            raise RuntimeError(
                "Failed to parse CB Insights API response as JSON."
            ) from e
        except Exception as e:
            logger.exception(
                f"Unexpected error while parsing CB Insights response: {e}"
            )
            raise RuntimeError(
                f"Unexpected error while parsing CB Insights response: {e}"
            ) from e

    def _format_sources(self, sources: list) -> str:
        """
        Format the sources array into a reference section.
        """
        if not sources:
            return ""

        formatted_sources = "\n\n## Sources\n"
        for source in sources:
            source_index = source.get("sourceIndex")
            result = source.get("result", {})
            title = result.get("title", "Unknown Source")
            url = result.get("url", "")
            date = result.get("date", "")

            if source_index is not None:
                formatted_sources += f"[^{source_index}]: {title}"
                if url:
                    formatted_sources += f" - {url}"
                if date:
                    formatted_sources += f" ({date})"
                formatted_sources += "\n"

        return formatted_sources
