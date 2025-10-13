"""Module containing the PegaExecutor for the integration of agents exposed over A2A."""

import asyncio
import logging
import os
from typing import Any, Callable, Dict
from uuid import uuid4

import httpx

from air.distiller.executor.executor import Executor
from air.types.distiller.executor.pega_config import PegaAgentConfig

logger = logging.getLogger(__name__)

try:
    from a2a.client import A2ACardResolver
    from a2a.types import AgentCard

except ImportError as exc:
    logger.error(
        "[Installation Failed] Missing A2A SDK dependencies. "
        "Install with: pip install '.[tah-a2a]'"
    )
    raise


class PegaExecutor(Executor):
    """
    Executor class for the Pega Agent.
    """

    agent_class: str = "PegaAgent"

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
        Initializes the PegaExecutor.
        """
        logger.debug(
            "Initializing PegaExecutor with role=%r, account=%r, project=%r, uuid=%r",
            role,
            account,
            project,
            uuid,
        )

        # Casting utility config to class-specific pydantic BaseModel
        pega_agent_config = PegaAgentConfig(**utility_config)

        # Retrieve required fields for authentication
        client_id = os.getenv(pega_agent_config.client_id, "")
        if not client_id:
            logger.info(
                "%s is not set in the environment.", pega_agent_config.client_id
            )
            raise RuntimeError("client_id is not set in the environment.")

        client_secret = os.getenv(pega_agent_config.client_secret, "")
        if not client_secret:
            logger.info(
                "%s is not set in the environment.", pega_agent_config.client_secret
            )
            raise RuntimeError("client_secret is not set in the environment.")

        token_url = pega_agent_config.token_url

        try:
            self._auth_token = self._get_token(client_id, client_secret, token_url)
            logger.info("Successfully retrieved authentication token.")
        except Exception as e:
            logger.exception("Failed to retrieve authentication token")
            raise RuntimeError(
                "Failed to fetch authentication token. Cannot continue."
            ) from e

        # Retrieve base url for retrieve agent card
        self._base_url = pega_agent_config.base_url

        # Retrieve wait_time for async client
        self._wait_time = pega_agent_config.wait_time

        self._session_id = str(uuid4())
        self._agent_card = None

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

    def _get_token(self, client_id: str, client_secret: str, token_url: str) -> str:
        """
        Get OAuth2 token using client credentials.
        """
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }

        with httpx.Client() as client:
            response = client.post(token_url, data=data)
            response.raise_for_status()

            token = response.json().get("access_token")
            if not token:
                raise RuntimeError("Token cannot be empty.")
            return token

    async def _execute_agent(self, **kwargs) -> str:
        """
        Executes the Pega Agent using a prompt.
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                "Missing 'prompt' parameter in PegaExecutor._execute_agent."
            )

        logger.debug("Running Pega Agent with prompt=%r", prompt)
        # Execute the asynchronous agent run in a synchronous context.
        return await self._process_agent_request(prompt)

    async def _process_agent_request(self, prompt: str) -> str:
        """
        Processes the agent request asynchronously.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self._auth_token,
        }
        async with httpx.AsyncClient(
            headers=headers, timeout=self._wait_time
        ) as client:
            try:
                if self._agent_card is None:
                    self._agent_card = await self._collect_agent_card(
                        client, self._base_url
                    )

                agent_card_url = self._agent_card.url
                response = await self._send_message(client, agent_card_url, prompt)

                response_text = await self._parse_response(response)
                return response_text

            except RuntimeError as e:
                logger.error("Failed to fetch the public agent card %s", e)
                raise
            except Exception:
                logger.exception("Failed to post message to the Pega agent.")
                raise

    async def _collect_agent_card(
        self, httpx_client: httpx.AsyncClient, base_url: str
    ) -> AgentCard:
        """
        Collect the A2A AgentCard.
        Uses the httpx client to instantiate an Agent Card Resolver (ACR).
        Then uses the ACR to collect the public agent card.
        """
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        try:
            logger.info("Attempting to fetch public agent card from: %s", base_url)
            public_card = await resolver.get_agent_card()
            logger.info("Successfully fetched public agent card:")
            logger.info(public_card.model_dump_json(indent=2, exclude_none=True))
            logger.info(
                "\nUsing PUBLIC agent card for client initialization (default)."
            )
            return public_card

        except Exception as e:
            logger.exception("Critical error fetching public agent card.")
            raise RuntimeError(
                "Failed to fetch the public agent card. Cannot continue."
            ) from e

    async def _send_message(
        self, client: httpx.AsyncClient, agent_card_url: str, prompt: str
    ) -> httpx.Response:
        """
        Post message to the Pega agent.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/send",
            "params": {
                "contextId": f"CTX-{self._session_id}",
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": prompt}],
                    "messageId": f"MSG-{uuid4()}",
                    "kind": "message",
                },
            },
        }

        try:
            response = await client.post(agent_card_url, json=payload)
            response.raise_for_status()

            logger.info("Pega agent response received")
            return response

        except httpx.RequestError as e:
            logger.error("Request error occurred posting message: %s", e)
            raise
        except Exception:
            logger.exception("Failed to retrieve response from Pega agent.")
            raise

    async def _parse_response(self, response: httpx.Response) -> str:
        """
        Retrieve plain text from Pega agent response.
        """
        try:
            response_text = response.json()["result"]["parts"][0]["text"]
            logger.info("Pega agent response received (len=%d)", len(response_text))
            return response_text
        except Exception as e:
            raise ValueError(
                f"Failed to retrieve plain text from agent response: {e}"
            ) from e
