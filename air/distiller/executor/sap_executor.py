"""Module containing the SAPExecutor for SAP Agent integration."""

import asyncio
import json
import logging
from typing import Any, Callable, Dict

import httpx

from air.distiller.executor.executor import Executor
from air.types.distiller.executor.sap_config import SAPAgentConfig

logger = logging.getLogger(__name__)


class SAPExecutor(Executor):
    """
    Executor class for SAP Agent.
    """

    agent_class: str = "SAPAgent"

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
        """Initializes the SAP Executor.

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
            "Initializing SAPExecutor with role=%r, account=%r, project=%r, uuid=%r",
            role,
            account,
            project,
            uuid,
        )

        # Casting utility config to class-specific pydantic BaseModel
        sap_agent_config = SAPAgentConfig(**utility_config)

        # Validate required fields in utility_config.
        self.url = sap_agent_config.url

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
        Executes the SAP agent using a prompt.
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                "Missing 'prompt' parameter in SAPExecutor._execute_agent."
            )

        logger.debug("Running SAP agent with prompt=%r", prompt)
        # Execute the asynchronous agent run in a synchronous context.
        return await self._process_agent_request(prompt)

    async def _process_agent_request(self, prompt: str) -> str:
        """Processes the agent request asynchronously."""

        response = await self.send_message(prompt)
        logger.info("SAP agent response received (length=%d)", len(response))

        return response

    async def send_message(self, prompt: str) -> str:
        # Create API request
        headers = {"Content-Type": "application/json"}

        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "id": "1",
            "params": {
                "message": {
                    "messageId": "msg-1",
                    "role": "user",
                    "parts": [{"type": "text", "text": prompt}],
                }
            },
        }

        # Send message asynchronously
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    follow_redirects=True,
                )
                response.raise_for_status()
            except httpx.RequestError:
                logger.exception("Failed to post message for SAP agent.")
                raise

        # Retrieve agent response
        response_text = ""
        try:
            # json_response = response.json()["result"]
            json_response = response.json()["result"]
            message_response = json_response.get(
                "Message", json_response.get("message", {})
            )
            response_text = message_response["parts"][0]["text"]
        except Exception:
            logger.exception("Failed to retrieve messages from SAP agent.")
            raise

        return response_text
