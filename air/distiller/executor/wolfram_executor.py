"""Module containing the WolframExecutor for Wolfram Alpha API integration."""

import asyncio
import logging
import os
from typing import Any, Callable, Dict

import httpx

from air.distiller.executor.executor import Executor
from air.types.distiller.executor.wolfram_config import WolframAgentConfig

logger = logging.getLogger(__name__)


class WolframExecutor(Executor):
    """
    Simplified Executor class for Wolfram Agent.
    Only handles API calls to Wolfram Alpha.
    Interpretation is handled by WolframInterpreterAgent in the backend.
    """

    agent_class: str = "WolframAgent"

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
        Initializes the simplified WolframExecutor.
        """
        logger.debug(
            "Initializing WolframExecutor with role=%r, account=%r, project=%r, uuid=%r",
            role,
            account,
            project,
            uuid,
        )

        self._wolfram_config = WolframAgentConfig(**utility_config)

        # Get Wolfram App ID from environment variable
        app_id_varname = self._wolfram_config.app_id
        self._app_id = os.getenv(app_id_varname, "")
        if self._app_id == "":
            error_msg = f"Wolfram App ID not set: {app_id_varname}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Initialize the base Executor with our specialized execution method
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
        Executes the Wolfram API call and returns raw response.
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                "Missing 'prompt' parameter in WolframExecutor._execute_agent."
            )

        logger.debug("Calling Wolfram API with prompt=%r", prompt)
        base_url = self._wolfram_config.base_url
        try:
            response = await self._call_wolfram_api(prompt, base_url)
            logger.info("Wolfram API response received (length=%d)", len(response))

            return response

        except Exception as e:
            logger.exception("Failed to call Wolfram API.")
            return f"Error calling Wolfram API: {str(e)}"

    async def _call_wolfram_api(self, query_text: str, wolfram_base_url: str) -> str:
        """
        Call the Wolfram Alpha API directly.
        """
        params = {
            "input": query_text,
            "appid": self._app_id,
        }

        timeout = self._wolfram_config.timeout
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url=wolfram_base_url, params=params)
                response.raise_for_status()

                # LLM API returns plain text response
                return response.text.strip()

        except httpx.RequestError as e:
            logger.error("Wolfram API request failed: %s", e)
            return f"Error querying Wolfram Alpha: {e}"
