"""Module containing the AmazonBedrockExecutor for AWS agent integration."""

import asyncio
import logging
import os
import uuid

from typing import Any, Callable, Dict

from air.distiller.executor.executor import Executor

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError as exc:
    logger.error(
        "[Installation Failed] Missing AWS Python SDK (boto3) dependencies. "
        "Install with: pip install boto3"
    )
    raise


class AmazonBedrockSessionManager:
    """
    Manages the AWS agent session lifecycle.
    """

    def __init__(
        self,
        client_key,
        client_secret,
        deployment_region,
        agent_id,
        alias_id,
        session_id,
    ):
        """
        Initializes the AmazonBedrockSessionManager.

        Args:
            client_key: The mapping to the variable that holds the AWS client key.
            client_secret: The mapping to the variable that holds the AWS client secret.
            deployment_region: The region where the AWS agent is deployed.
            agent_id: The identifier of the created AWS agent.
            alias_id: The alias identifier  for the AWS agent.
            session_id: The identifier of the session where the messages will be posted.
        """
        self.client_key = client_key
        self.client_secret = client_secret
        self.deployment_region = deployment_region
        self.agent_id = agent_id
        self.alias_id = alias_id
        self.session_id = session_id

        self.initialize_client()

    def initialize_client(self):
        try:
            # Initialize the AWS project client using the provided credentials.
            self.client = boto3.client(
                service_name="bedrock-agent-runtime",
                region_name=self.deployment_region,
                aws_access_key_id=os.getenv(self.client_key),
                aws_secret_access_key=os.getenv(self.client_secret),
            )

            logger.info("Successfully initialized AWS client.")
        except Exception:
            logger.exception("Failed to initialize AWS client")
            raise

    def initiate_session(self):
        """Creates a new session for the AWS agent"""
        self.session_id = str(uuid.uuid4()).replace("-", "")
        logger.info("Successfully created new session %s.", self.session_id)

    def send_message(self, prompt):
        """
        Sends a message to the AWS agent.
        """
        try:
            response = self.client.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.alias_id,
                enableTrace=True,
                sessionId=self.session_id,
                inputText=prompt,
                streamingConfigurations={
                    "applyGuardrailInterval": 20,
                    "streamFinalResponse": False,
                },
            )

            completion = ""
            for event in response.get("completion"):
                # Collect agent output.
                if "chunk" in event:
                    chunk = event["chunk"]
                    completion += chunk["bytes"].decode()

                # Log trace output.
                if "trace" in event:
                    trace_event = event.get("trace")
                    trace = trace_event["trace"]
                    for key, value in trace.items():
                        logging.info("%s: %s", key, value)

            return completion
        except ClientError as e:
            print(f"Client error: {str(e)}")
            logger.error("Client error: %s", {str(e)})


class AmazonBedrockExecutor(Executor):
    """Executor class for AWS Agents leveraging AWS Bedrock.

    This class extends the generic `Executor` to provide functionality for interacting
    with an AWS Bedrock agent created in the UI.
    """

    agent_class: str = ""

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
        """Initializes the AWS Executor.

        Args:
            func: A dictionary mapping function names to callables.
            send_queue: An asyncio.Queue for sending output messages.
            account: The account identifier.
            project: The project identifier.
            uuid: A unique identifier for the session or request.
            role: The role identifier for this executor (e.g., "agent").
            utility_config: A configuration dictionary containing:
                - "client_key": Mapping to the env var holding the AWS Client Key
                - "client_secret": apping to the env var holding the AWS Client Secret
                - "deployment_region": Deployment region for the AWS Agent
                - "agent_id": The identifier for the AWS agent.
                - "alias_id": The identifier for the AWS agent alias (version).
                - "session_id": The identifier of the session where the messages will be posted (optional)
            return_string: Flag to determine if the result should be returned as a string.

        Raises:
            ValueError: If any required configuration key is missing.
            Exception: If initialization of the AWS session manager or agent fails.
        """
        logger.debug(
            "Initializing AWS Executor with role=%r, account=%r, project=%r, uuid=%r",
            role,
            account,
            project,
            uuid,
        )

        # Validate required fields in utility_config.
        client_key_varname = utility_config.get("client_key")
        client_secret_varname = utility_config.get("client_secret")
        deployment_region = utility_config.get("deployment_region")
        agent_id = utility_config.get("agent_id")
        alias_id = utility_config.get("alias_id")
        session_id = utility_config.get("session_id", "")

        if not client_key_varname:
            raise ValueError(
                "Missing 'client_key' in utility_config for AmazonBedrockExecutor."
            )
        if not client_secret_varname:
            raise ValueError(
                "Missing 'client_secret' in utility_config for AmazonBedrockExecutor."
            )
        if not deployment_region:
            raise ValueError(
                "Missing 'deployment_region' in utility_config for AmazonBedrockExecutor."
            )
        if not agent_id:
            raise ValueError(
                "Missing 'agent_id' in utility_config for AmazonBedrockExecutor."
            )
        if not alias_id:
            raise ValueError(
                "Missing 'alias_id' in utility_config for AmazonBedrockExecutor."
            )

        try:
            # Initialize AWS session manager
            self.session_manager = AmazonBedrockSessionManager(
                client_key=client_key_varname,
                client_secret=client_secret_varname,
                deployment_region=deployment_region,
                agent_id=agent_id,
                alias_id=alias_id,
                session_id=session_id,
            )
            logger.info("Successfully initialized AWS session manager.")
        except Exception:
            logger.exception("Failed to initialize AWS session manager.")
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
        Executes the AWS agent using a prompt.
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                "Missing 'prompt' parameter in AmazonBedrockExecutor._execute_agent."
            )

        logger.debug("Running AWS agent with prompt=%r", prompt)
        # Execute the asynchronous agent run in a synchronous context.
        return asyncio.run(self._process_agent_request(prompt))

    async def _process_agent_request(self, prompt: str) -> str:
        """Processes the agent request asynchronously."""
        if not self.session_manager.session_id:
            # Create a new session with the Amazon Bedrock agent.
            self.session_manager.initiate_session()
        else:
            # Using session specified in config
            logger.info(
                "Using session %s as specified in config.",
                self.session_manager.session_id,
            )

        try:
            # Post the user prompt as a message within the conversation session.
            response = self.session_manager.send_message(prompt=prompt)
        except Exception:
            logger.exception("Failed to post message for AWS agent.")
            raise
        return str(response)
