"""Module containing the SalesforceExecutor for Salesforce Agentfoce integration."""

import asyncio
import logging
import uuid
from typing import Any, Callable, Dict
import os
import requests

from air.distiller.executor.executor import Executor


logger = logging.getLogger(__name__)


class SalesforceSessionManager:
    """
    Manages the Salesforce agent session lifecycle.
    """

    def __init__(self, domain, agent_id, token):
        """
        Initializes the SalesforceSessionManager.

        Args:
            domain: The Salesforce orgfarm domain URL.
            agent_id: The identifier of the created Salesforce agent.
            token: The access token for the Salesforce agent.
        """
        self.domain = domain
        self.agent_id = agent_id
        self.token = token

    @classmethod
    def get_access_token(cls, client_key_var, client_secret_var, domain):
        """
        Retrieves an access token for the Salesforce agent.
        """
        consumer_key = os.getenv(client_key_var)
        consumer_secret = os.getenv(client_secret_var)

        url = f"https://{domain}/services/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": consumer_key,
            "client_secret": consumer_secret,
        }

        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()  # Raises error for 4xx/5xx responses
        token_info = response.json()
        token = token_info.get("access_token")
        return token

    def initiate_session(self):
        """
        Initiates a new Salesforce agent session.
        """
        session_id = str(uuid.uuid4())
        sequence_id = 0

        url = f"https://api.salesforce.com/einstein/ai-agent/v1/agents/{self.agent_id}/sessions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        payload = {
            "externalSessionKey": session_id,
            "instanceConfig": {"endpoint": f"https://{self.domain}"},
            "streamingCapabilities": {"chunkTypes": ["Text"]},
            "bypassUser": True,
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        session_info = response.json()
        return (
            session_info["sessionId"],
            sequence_id,
            session_info["messages"][0]["message"],
        )

    def send_message(self, session_id, sequence_id, message_text):
        """
        Sends a message to the Salesforce agent.
        """
        url = f"https://api.salesforce.com/einstein/ai-agent/v1/sessions/{session_id}/messages"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        payload = {
            "message": {"sequenceId": sequence_id, "type": "Text", "text": message_text}
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        sequence_id += 1  # Increment for next call
        return response_data


class SalesforceExecutor(Executor):
    """
    Executor class for Salesforce Agent.
    """

    agent_class: str = "SalesforceAgent"

    # pylint: disable=too-many-arguments,too-many-locals
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
        Initializes the SalesforceExecutor.
        """
        logger.debug(
            "Initializing SalesforceExecutor with role=%r, account=%r, project=%r, uuid=%r",
            role,
            account,
            project,
            uuid,
        )

        # Validate required fields in utility_config.
        client_key_varname = utility_config.get("client_key")
        client_secret_varname = utility_config.get("client_secret")
        self.domain = utility_config.get("domain")
        self.agent_id = utility_config.get("agent_id")
        if not self.domain:
            raise ValueError(
                "Missing 'domain' in utility_config for SalesforceExecutor."
            )
        if not self.agent_id:
            raise ValueError(
                "Missing 'agent_id' in utility_config for SalesforceExecutor."
            )

        try:
            # Initialize connection by retrieving the authentication token
            self.session_manager = SalesforceSessionManager(
                domain=self.domain,
                agent_id=self.agent_id,
                token=SalesforceSessionManager.get_access_token(
                    client_key_varname, client_secret_varname, self.domain
                ),
            )
            logger.info("Successfully retrieved connection token.")
        except Exception:
            logger.exception("Failed to retrieve connection token")
            raise

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

    def _execute_agent(self, **kwargs) -> str:
        """
        Executes the Salesforce agent using a prompt.
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                "Missing 'prompt' parameter in SalesforceExecutor._execute_agent."
            )

        logger.debug("Running Salesforce agent with prompt=%r", prompt)
        # Execute the asynchronous agent run in a synchronous context.
        return asyncio.run(self._process_agent_request(prompt))

    async def _process_agent_request(self, prompt: str) -> str:
        """Processes the agent request asynchronously."""
        try:
            # Create a new session with the Salesfoce agent.
            session_id, sequence_id, welcome_msg = (
                self.session_manager.initiate_session()
            )
            logger.info("Successfully created new session %s.", session_id)
        except Exception:
            logger.exception("Failed to create session with the Salesforce agent.")
            raise

        try:
            # Post the user prompt as a message within the conversation session.
            response = self.session_manager.send_message(
                session_id=session_id, sequence_id=sequence_id, message_text=prompt
            )
        except Exception:
            logger.exception("Failed to post message for Salesforce agent.")
            raise

        try:
            # Retrieve agent response
            message = response["messages"][0]["message"]
        except Exception:
            logger.exception("Failed to retrieve messages for Salesforce agent.")
            raise

        # Combine all collected text parts into a single response string.
        final_response = message
        logger.info(
            "Salesforce agent response received (length=%d)", len(final_response)
        )

        return final_response
