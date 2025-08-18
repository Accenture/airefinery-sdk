"""Module containing the GoogleExecutor for Vertex AI integration."""

import asyncio
import logging
import json
from typing import Any, Callable, Dict, Optional

from air.distiller.executor.executor import Executor
from air.types.distiller.executor.google_config import GoogleAgentConfig

logger = logging.getLogger(__name__)

try:
    import vertexai
    from google.genai import types

    # The following import is necessary as vertexai by default doesn't load
    # agent_engines
    from vertexai import agent_engines  # pylint:disable=unused-import
except ImportError as exc:
    logger.error(
        "[Installation Failed] Missing Vertex AI SDK dependencies. "
        'Install with: pip install "airefinery-sdk[tah-vertex-ai]"'
    )
    raise


class GoogleExecutor(Executor):
    """Executor class for GoogleAgent leveraging Vertex AI Agents.

    This class extends the generic `Executor` to provide functionality
    for interacting with Google Vertex AI's agent engines. It requires a
    valid `resource_name` in `utility_config` to fetch a Vertex AI agent engine.
    """

    agent_class: str = "GoogleAgent"

    # pylint: disable=too-many-arguments,too-many-positional-arguments
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
        """Initializes the GoogleExecutor.

        Args:
            func: A dictionary mapping function names to callables.
            send_queue: An asyncio queue for sending output.
            account: The account identifier (if applicable).
            project: The project identifier for Vertex AI and logging.
            uuid: A user or session UUID.
            role: The role name or identifier for this executor (e.g., "agent").
            utility_config: A configuration dictionary. Must include
                "resource_name" to load the agent.
            return_string: Whether to return a stringified output.

        Raises:
            ValueError: If "resource_name" is not present in `utility_config`.
            Exception: If any other error occurs during initialization.
        """
        logger.debug(
            "Initializing GoogleExecutor with role=%r, account=%r, project=%r, uuid=%r",
            role,
            account,
            project,
            uuid,
        )

        google_config = GoogleAgentConfig(**utility_config)

        # Retrieve required fields from utility_config.
        resource_name = google_config.resource_name

        # Fetch agent engine
        try:
            self.agent_engine = vertexai.agent_engines.get(resource_name)
            logger.info("Successfully loaded Vertex AI agent engine: %s", resource_name)
        except Exception:
            logger.exception(
                "Failed to acquire Vertex AI agent engine: %s", resource_name
            )
            raise

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

    def _execute_agent(self, **kwargs) -> str:
        """Executes the Vertex AI agent engine using prompt and session_id.

        This method is passed to the parent Executor, allowing external
        triggers to invoke the agent.

        Args:
            **kwargs: Expected to contain:
                - prompt (str): The prompt or query for the agent.
                - session_id (str): The session identifier for the agent.
                - user_id (str): The uuid from the current session.

        Returns:
            A string response from the Vertex AI agent engine.

        Raises:
            ValueError: If the prompt is not provided.
            Exception: If agent execution fails unexpectedly.
        """
        prompt = kwargs.get("prompt")
        session_id = kwargs.get("session_id", "")
        user_id = kwargs.get("uuid", "user_id")

        if not prompt:
            raise ValueError(
                "Missing 'prompt' parameter in GoogleExecutor._execute_agent."
            )

        logger.debug("Running agent with prompt=%r, session_id=%r", prompt, session_id)

        try:
            request = {
                "user_id": user_id,
                "session_id": session_id,
                "message": {"role": "user", "parts": [{"text": prompt}]},
                "events": [],
                "artifacts": [],
                "authorizations": {},
            }
            request_json = json.dumps(request)
        except Exception:
            logger.exception("Error while running Vertex AI agent engine.")
            raise

        result_parts = []
        # Linter does not recognize streaming_agent_run_with_events as a method of AgentEngine.
        # This is seems like a persistent issue from Google's side.
        # Previously: Linter does not recognize run as a method of AgentEngine.
        try:
            for (
                stream_chunk
            ) in self.agent_engine.streaming_agent_run_with_events(  # pylint:disable=no-member
                request_json=request_json
            ):
                try:
                    events = stream_chunk.get("events", [])
                    for event in events:
                        try:
                            parts = event.get("content", {}).get("parts", [])
                            for part in parts:
                                text_chunk = part.get("text", "")
                                if text_chunk:
                                    result_parts.append(text_chunk)
                        except Exception as e:
                            # Log or handle error in processing an event
                            logger.exception(f"Error processing event: {e}")
                except Exception as e:
                    # Log or handle error in processing a stream chunk
                    logger.exception(f"Error processing stream chunk: {e}")
        except Exception as e:
            # Log or handle error in the overall stream
            logger.exception(f"Error during streaming: {e}")

        final_response = "".join(result_parts)
        logger.info("Agent response received (length=%d)", len(final_response))

        return final_response
