"""Module containing the AzureExecutor for Azure AI Web agent integration."""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from air.distiller.executor.executor import Executor

logger = logging.getLogger(__name__)

try:
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential
except ImportError as exc:
    logger.error(
        "[Installation Failed] Missing Azure AI SDK dependencies. "
        'Install with: pip install "airefinery-sdk[tah-azure-ai]"'
    )
    raise


class AzureExecutor(Executor):
    """Executor class for Azure Web Agent leveraging Azure AI's agent engines.

    This class extends the generic `Executor` to provide functionality for interacting
    with an Azure AI agent via its Web UI creation mode.
    """

    agent_class: str = "AzureAIAgent"

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
        """Initializes the AzureExecutor.

        Args:
            func: A dictionary mapping function names to callables.
            send_queue: An asyncio.Queue for sending output messages.
            account: The account identifier.
            project: The project identifier.
            uuid: A unique identifier for the session or request.
            role: The role identifier for this executor (e.g., "agent").
            utility_config: A configuration dictionary containing:
                - "agent_id": The identifier for the Azure agent.
                - "connection_string": The connection string for the Azure project.
            return_string: Flag to determine if the result should be returned as a string.

        Raises:
            ValueError: If any required configuration key is missing.
            Exception: If initialization of the Azure client or agent fails.
        """
        logger.debug(
            "Initializing AzureExecutor with role=%r, account=%r, project=%r, uuid=%r",
            role,
            account,
            project,
            uuid,
        )

        # Validate required fields in utility_config.
        agent_id = utility_config.get("agent_id")
        connection_string = utility_config.get("connection_string")
        if not agent_id:
            raise ValueError("Missing 'agent_id' in utility_config for AzureExecutor.")
        if not connection_string:
            raise ValueError(
                "Missing 'connection_string' in utility_config for AzureExecutor."
            )

        try:
            # Initialize the Azure project client using the provided connection string
            # and default credentials.
            self.project_client = AIProjectClient.from_connection_string(  # type: ignore[attr-defined]
                credential=DefaultAzureCredential(),
                conn_str=connection_string,
            )

            # Retrieve the agent using its unique agent_id.
            self.agent = self.project_client.agents.get_agent(agent_id)
            logger.info("Successfully loaded Azure agent: %s", agent_id)
        except Exception:
            logger.exception("Failed to load Azure agent with agent_id: %s", agent_id)
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
        """Executes the Azure agent using a prompt.

        This method is passed to the parent Executor and is responsible for triggering
        the Azure agent's processing using a synchronous call to an asynchronous function.

        Args:
            **kwargs: Expected to contain:
                - prompt (str): The prompt or query for the agent.

        Returns:
            A string response from the Azure agent.

        Raises:
            ValueError: If the 'prompt' parameter is not provided.
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                "Missing 'prompt' parameter in AzureExecutor._execute_agent."
            )

        logger.debug("Running Azure agent with prompt=%r", prompt)
        # Execute the asynchronous agent run in a synchronous context.
        return asyncio.run(self._run_agent(prompt))

    async def _run_agent(self, prompt: str) -> str:
        """Runs the Azure agent in Web UI mode asynchronously.

        This method creates a new conversation thread, sends the user prompt,
        processes the run, collects the assistant responses, and then cleans up by deleting
        the thread.

        Args:
            prompt (str): The prompt or query for the agent.

        Returns:
            A string containing the concatenated assistant responses.

        Raises:
            Exception: Propagates any exception encountered during thread creation,
                       message posting, run processing, or message retrieval.
        """
        try:
            # Create a new conversation thread using the Web UI method.
            thread = self.project_client.agents.create_thread()
        except Exception:
            logger.exception("Failed to create thread for Azure agent.")
            raise

        try:
            # Post the user prompt as a message within the conversation thread.
            self.project_client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=prompt,
            )
        except Exception:
            logger.exception("Failed to post message for Azure agent.")
            raise

        try:
            # Initiate processing of the run using the Azure agent via the Web UI.
            run = self.project_client.agents.create_and_process_run(
                thread_id=thread.id, agent_id=self.agent.id
            )
        except Exception:
            logger.exception("Failed to process run for Azure agent.")
            raise

        try:
            # Retrieve all messages associated with the conversation thread.
            messages = self.project_client.agents.list_messages(thread_id=thread.id)
        except Exception:
            logger.exception("Failed to retrieve messages for Azure agent.")
            raise

        # Iterate over the messages and collect text parts from assistant responses.
        response_parts = []
        for msg in messages.get("data", []):
            if msg.get("role") == "assistant" and msg.get("content"):
                for part in msg["content"]:
                    if part.get("type") == "text":
                        # Extract the text value from the message part.
                        text_value = part.get("text", {}).get("value", "")
                        if text_value:
                            response_parts.append(text_value.strip())
        # Combine all collected text parts into a single response string.
        final_response = "\n".join(response_parts)
        logger.info("Azure agent response received (length=%d)", len(final_response))

        # Attempt to clean up by deleting the conversation thread.
        try:
            self.project_client.agents.delete_thread(thread.id)
        except Exception as e:
            logger.warning("Failed to delete thread %s: %s", thread.id, str(e))

        return final_response
