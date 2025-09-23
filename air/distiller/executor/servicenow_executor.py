"""Module containing the ServiceNowExecutor for ServiceNow Agent integration."""

# pylint: disable=no-member

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Callable, Dict, Tuple

import httpx
import requests

from air.distiller.executor.executor import Executor
from air.types.distiller.executor.servicenow_config import (
    ServiceNowAgentConfig,
)

logger = logging.getLogger(__name__)


class ServiceNowExecutor(Executor):
    """
    Executor class for ServiceNow Agent.
    """

    agent_class: str = "ServiceNowAgent"

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
        """Initializes the ServiceNow Executor.

        Args:
            func: A dictionary mapping function names to callables.
            send_queue: An asyncio.Queue for sending output messages.
            account: The account identifier.
            project: The project identifier.
            uuid: A unique identifier for the session or request.
            role: The role identifier for this executor (e.g., "agent").
            utility_config: A configuration dictionary.
            return_string: Flag to determine if the result should be returned as a string.

        Raises:
            ValueError: If any required configuration key is missing.
        """
        logger.debug(
            "Initializing ServiceNowExecutor with role=%r, account=%r, project=%r, uuid=%r",
            role,
            account,
            project,
            uuid,
        )

        # Casting utility config to class-specific pydantic BaseModel
        self._servicenow_agent_config = ServiceNowAgentConfig(**utility_config)

        # Retrieve config fields
        self._servicenow_token = os.getenv(
            self._servicenow_agent_config.servicenow_token
        )

        if self._servicenow_token is None:
            logger.warning(
                "ServiceNow token environment variable '%s' is not set. "
                "Please set the environment variable to authenticate with ServiceNow.",
                self._servicenow_agent_config.servicenow_token,
            )

        self._headers = {
            "Content-Type": "application/json",
            "x-sn-apikey": self._servicenow_token,
        }
        agent_card = self._servicenow_agent_config.agent_card
        self._public_agent_card_path = agent_card.public.public_agent_card_path
        self._rpc_url = agent_card.public.rpc_url
        self._wait_time = self._servicenow_agent_config.wait_time

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

    async def _execute_agent(self, **kwargs) -> str:
        """
        Executes the ServiceNow agent using a prompt.
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                "Missing 'prompt' parameter in ServiceNowExecutor._execute_agent."
            )
        logger.debug("Running ServiceNow agent with prompt=%r", prompt)

        return await self._process_agent_request(prompt)

    async def _process_agent_request(self, prompt: str) -> str:
        """
        Processes the agent request asynchronously.
        """
        response = await self.send_message(prompt)
        response_text = self.parse_response(response)
        logger.info(
            "ServiceNow agent response received (length=%d)", len(response_text)
        )

        return response_text

    async def send_message(self, prompt: str) -> httpx.Response:
        """
        Sends the query to the ServiceNow agent.
        """
        # Retrieve post url and create body of the API request
        agent_url = await self.retrieve_agent_url()
        request_body = self.create_request_body(prompt)

        # Post request
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._wait_time)
            ) as client:
                response = await client.post(
                    agent_url,
                    json=request_body,
                    headers=self._headers,
                )
                response.raise_for_status()
        except httpx.RequestError as e:
            logger.error(f"Failed to post message to the ServiceNow agent: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

        return response

    async def retrieve_agent_url(self) -> str:
        """
        Retrieves the agent_url from the public agent card.
        The agent_url is used to post the requests to the ServiceNow agent.
        """
        final_agent_card_to_use = await self.collect_agent_card()
        try:
            if not "url" in final_agent_card_to_use:
                raise ValueError("Missing 'url' in the agent card.")

            agent_url = final_agent_card_to_use.get("url", "")
            if agent_url == "":
                raise ValueError("'agent_url' should not be empty!")
        except Exception as e:
            raise ValueError(f"Failed to extract 'url' from agent card: {e}") from e

        return agent_url

    async def collect_agent_card(self) -> dict:
        """
        Retrieves the A2A AgentCard of ServiceNow Agent.
        """
        final_agent_card_to_use = {}
        try:
            agent_card_url = f"{self._rpc_url}/{self._public_agent_card_path}"
            logger.info(f"Attempting to fetch public agent card from: {agent_card_url}")
            async with httpx.AsyncClient() as client:
                response = await client.get(agent_card_url, headers=self._headers)
                response.raise_for_status()
                final_agent_card_to_use = response.json()
        except httpx.RequestError as e:
            logger.error(f"Error retrieving agent card: {e}")
            raise RuntimeError("Failed to retrieve agent card.") from e

        if final_agent_card_to_use is None:
            raise RuntimeError("No agent card was successfully retrieved.")

        return final_agent_card_to_use

    def create_request_body(self, prompt: str) -> dict:
        """
        Creates the request body for the ServiceNow agent.
        """
        message_id = str(uuid.uuid4()).replace("-", "")
        request_id = str(uuid.uuid4()).replace("-", "")

        request_body = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "kind": "message",
                    "parts": [
                        {
                            "kind": "text",
                            "text": prompt,
                        }
                    ],
                    "messageId": message_id,
                    "contextId": None,
                    "taskId": None,
                },
                "metadata": {},
                "pushNotificationUrl": "",
            },
            "id": request_id,
        }

        return request_body

    def parse_response(self, response) -> str:
        """
        Parses the response of the ServiceNow agent.
        """
        try:
            response_dict = json.loads(response.text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in ServiceNow response: {e}") from e

        # Check if the required fields exist in the dictionary:
        if "result" not in response_dict:
            raise ValueError("Missing 'result' field in ServiceNow response")
        result = response_dict["result"]

        if "status" not in result:
            raise ValueError("Missing 'status' field in ServiceNow response result")
        status = result["status"]

        if "message" not in status:
            raise ValueError("Missing 'message' field in ServiceNow response status")
        message = status["message"]

        if "parts" not in message:
            raise ValueError("Missing 'parts' field in ServiceNow response message")
        parts = message["parts"]
        if not isinstance(parts, list) or len(parts) == 0:
            raise ValueError(
                "'parts' field must be a non-empty list in ServiceNow response"
            )

        # Get the first part as it contains the textual response of the agent
        part = parts[0]
        if "text" not in part:
            raise ValueError("Missing 'text' field in ServiceNow response message part")
        response_text = part["text"]
        if not isinstance(response_text, str):
            raise ValueError("'text' field must be a string in ServiceNow response")

        return response_text
