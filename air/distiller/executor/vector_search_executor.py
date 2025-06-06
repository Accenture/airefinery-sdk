"""
Executor for custom ai search retrievers
"""

import asyncio
import logging
from typing import Any, Callable, Dict

from air.distiller.executor.executor import Executor

# Set up logging
logger = logging.getLogger(__name__)


class CustomVectorSearchExecutor(Executor):
    """Executor class for ResearchAgent.

    Extends Executor to support multiple retriever functions based on retriever types.

    """

    agent_class: str = "ResearchAgent"

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
        """Initialize the CustomVectorSearchExecutor.

        Args:
            func (Dict[str, Callable]): A dictionary mapping retriever types to callables.
            send_queue (asyncio.Queue): Queue to send output to.
            account (str): Account identifier.
            project (str): Project identifier.
            uuid (str): User UUID.
            role (str): Role of the executor (typically the agent name).
            utility_config (Dict[str, Any]): Configuration dictionary for utility agents.
            return_string (bool): Whether to return a stringified output back.

        Raises:
            ValueError: If an unsupported executor type is
                        specified or required configuration is missing.
            Exception: For any other errors during initialization.
        """
        # Initialize func as a dictionary of callables.
        # Perform setup based on retriever type specified in utility_config.
        self.func = {}
        try:
            retriever_config_list = utility_config.get("retriever_config_list")
            if not retriever_config_list:
                error_msg = "retriever_config_list is missing in utility_config."
                logger.error(error_msg)
                raise ValueError(error_msg)

            for retriever_config in retriever_config_list:

                retriever_class = retriever_config.get("retriever_class")

                if not retriever_class:
                    error_msg = "retriever_config.retriever_class is missing in retriever_config."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                retriever_name = retriever_config.get("retriever_name")
                if not retriever_name:
                    error_msg = "retriever_config.retriever_name is missing in retriever_config."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                if retriever_class == "CustomRetriever":
                    if retriever_name not in func:
                        error_msg = (
                            f"No executor for custom retriever {retriever_name} found. "
                            + "You must create this executor and add it to your "
                            + "executor_dict."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    else:
                        self.func[retriever_name] = func[retriever_name]
                        logger.info(
                            f"{retriever_name} already exists in func; using the user-defined one."
                        )

        except Exception as e:
            logger.exception(
                "Error occurred during ResearchAgent retriever initialization."
            )
            # Re-raise the exception to indicate failure during initialization
            raise e

        return_string = False
        # Initialize the base class with the func dictionary
        super().__init__(
            func=self.func,
            send_queue=send_queue,
            account=account,
            project=project,
            uuid=uuid,
            role=role,
            return_string=return_string,
        )

    async def __call__(self, request_id: str, *args, **kwargs):
        """Execute the appropriate retriever function based on executor.

        Args:
            request_id (str): Unique identifier for the request.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of executing the selected retriever function.

        Raises:
            ValueError: If 'executor' is not specified or invalid.
            TypeError: If 'executor' is not a string.
        """
        executor = kwargs.pop("__executor__", None)
        if executor is None:
            error_msg = "'__executor__' must be specified in kwargs."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(executor, str):
            error_msg = f"'__executor__'  must be a string, got {type(executor)}."
            logger.error(error_msg)
            raise TypeError(error_msg)
        if executor not in self.func:
            error_msg = f"Retriever type '{executor}' is not available in func."
            logger.error(error_msg)
            raise ValueError(error_msg)

        selected_func = self.func[executor]
        logger.debug(f"Executing retriever function for type: {executor}")

        # Call the base class __call__ method with the selected function
        return await super().__call__(
            request_id=request_id, func=selected_func, *args, **kwargs
        )

    def validate_result(self, result):
        """
        Validates that result is alist and each item in the result is a dictionary
        with a 'result' key containing a string and a 'score' key containing a numeric value.

        Args:
            result_list (list): List of dictionaries to validate.

        Returns:
            bool: True if all dictionaries in the list are valid, False otherwise.
        """

        error_message = "Incorrect results format."
        error_message += (
            "\nThe result from the CustomRetriever must be a list of dictionaries."
        )
        error_message += "\nEach dictionary must be in the following format: {'result': 'a string result', 'score': floating point score}"

        if not isinstance(result, list):
            logger.error(error_message)
            return []

        for item in result:
            # Check if the item is a dictionary
            if not isinstance(item, dict):
                logger.error(error_message)
                return []

            # Check if 'result' key exists and is a string
            if "result" not in item or not isinstance(item["result"], str):
                logger.error(error_message)
                return []

            # Check if 'score' key exists and is a numeric type (int or float)
            if "score" not in item or not isinstance(item["score"], (int, float)):
                logger.error(error_message)
                return []

        return result
