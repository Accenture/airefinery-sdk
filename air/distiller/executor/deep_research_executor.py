"""
Module: deep_research_executor.py

This module defines the DeepResearch Executor class which delegates to HumanExecutor.
It constructs a HumanExecutor and forwards calls to it.

"""

import asyncio
import logging
from typing import Any, Callable, Dict, Union

from air.distiller.executor.executor import Executor
from air.distiller.executor.human_executor import HumanExecutor

logger = logging.getLogger(__name__)


class DeepResearchExecutor(Executor):
    """
    A thin adapter that delegates all work to a HumanExecutor.
    """

    agent_class: str = "DeepResearchAgent"

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
        human_cfg_dict = utility_config.get("human_agent_config", utility_config)

        # Build the underlying HumanExecutor
        self.human_executor = HumanExecutor(
            func=func,
            send_queue=send_queue,
            account=account,
            project=project,
            uuid=uuid,
            role=role,
            utility_config=human_cfg_dict,
            return_string=return_string,
        )

        # Expose common attributes for compatibility / tracing
        # All queueing and payload formatting are handled by HumanExecutor.
        self.account = account
        self.project = project
        self.uuid = uuid
        self.role = role
        self.send_queue = send_queue
        self.return_string = return_string

    async def __call__(self, request_id: str, *args, **kwargs):
        """
        Forwards the call to the underlying HumanExecutor.

        Args:
            request_id (str): Unique identifier for the current request.
            *args, **kwargs: Passed through to HumanExecutor.__call__.
                Recognized kwarg: 'query' (str) – the prompt to display.

        Returns:
            str: The collected user feedback from the HumanExecutor.
        """
        logger.info(
            "DeepResearchExecutor delegating to HumanExecutor for request_id=%s",
            request_id,
        )
        return await self.human_executor(request_id, *args, **kwargs)
