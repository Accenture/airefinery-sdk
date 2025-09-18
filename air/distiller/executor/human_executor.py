"""
Module: human_executor.py

This module defines the HumanExecutor class, which acts as an interface between
a human user and agent in interactive systems. When a 'wait' status is received,
this executor prompts the user for input and returns the response.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Union

from air.distiller.executor.executor import Executor
from air.types.distiller.client import (
    DistillerMessageRequestType,
    DistillerOutgoingMessage,
    DistillerMessageRequestArgs,
)
from air.types.distiller.executor.human_config import HumanAgentConfig

logger = logging.getLogger(__name__)


async def input_method_from_terminal(query: str) -> str:
    """
    Collects string input from the terminal asynchronously.

    Args:
        query (str): The prompt to display to the user.

    Returns:
        str: The collected user's input from terminal as a string.
    """
    loop = asyncio.get_running_loop()
    content = await loop.run_in_executor(None, input, query)
    return content


class HumanExecutor(Executor):
    """
    Executor for the HumanAgent. Prompts the user for input when a 'wait' status is received.

    This class determines how the input is collected (via terminal or a custom function),
    and sends the user’s response back through an asyncio queue to downstream agents.
    """

    agent_class: str = "HumanAgent"

    def __init__(
        self,
        func: Union[Callable, Dict[str, Callable]],
        send_queue: asyncio.Queue,
        account: str,
        project: str,
        uuid: str,
        role: str,
        utility_config: Dict[str, Any],
        return_string: bool = True,
    ):
        """
        Initializes the HumanExecutor with configuration for user input handling.

        Args:
            func (Union[Callable, Dict[str, Callable]]): The input collection function.
                Required if using 'Custom' mode.
            send_queue (asyncio.Queue): Queue to send the user response to downstream components.
            account (str): Identifier for the user or organization.
            project (str): Identifier for the current project.
            uuid (str): Unique identifier for the request.
            role (str): Role name associated with this executor instance.
            utility_config (Dict[str, Any]): Configuration dictionary for HumanAgent behavior.
            return_string (bool, optional): Whether to return the response as a raw string. Defaults to True.

        Raises:
            ValueError: If a required custom function is not provided or is invalid.
        """

        # Casting utility config to class-specific pydantic BaseModel
        human_agent_config = HumanAgentConfig(**utility_config)

        if human_agent_config.user_input_method == "Terminal":
            self.input_func = input_method_from_terminal
        elif human_agent_config.user_input_method == "Custom":
            if func == {}:
                raise ValueError(
                    "In 'Custom' config, an input function must be defined in the human executor."
                )
            else:
                if not callable(func):
                    raise ValueError(
                        "In 'Custom' config, the provided `func` must be a callable"
                    )
                self.input_func = func
        else:
            raise ValueError(
                f"{human_agent_config.user_input_method} mode is not supported."
            )

        super().__init__(
            func=self.input_func,
            send_queue=send_queue,
            account=account,
            project=project,
            uuid=uuid,
            role=role,
            return_string=return_string,
        )

    async def __call__(self, request_id: str, *args, **kwargs):
        """
        Invokes the input collection process and sends the result to the queue.

        Returns:
            str: The collected user feedback.

        Args:
            request_id (str): Unique identifier for the current request.
            *args: Additional unused arguments.
            **kwargs: Should contain 'query' for the user prompt (optional).
        """

        logger.info("HumanExecutor called for request_id: %s", request_id)
        query = kwargs.get("query", "Please provide your input:")
        prompt_message = f"{query}\n> "

        logger.info("Prompting user: '%s'", prompt_message.strip())
        user_input = await self.input_func(prompt_message)
        logger.info("User entered: '%s'", user_input)

        response_request_args = DistillerMessageRequestArgs(content=user_input)
        response_payload = DistillerOutgoingMessage(
            account=self.account,
            project=self.project,
            uuid=self.uuid,
            role=self.role,
            request_id=request_id,
            request_type=DistillerMessageRequestType.EXECUTOR,
            request_args=response_request_args,
        )

        logger.info("Sending response payload for request_id: %s", request_id)
        await self.send_queue.put(response_payload)
        return user_input
