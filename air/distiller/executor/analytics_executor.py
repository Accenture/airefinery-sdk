import asyncio
from typing import Callable, Dict, Any
import logging

from air.distiller.executor.executor import Executor
from air.api import PostgresAPI, PandasAPI

# Set up logging
logger = logging.getLogger(__name__)


class AnalyticsExecutor(Executor):
    """Executor class for AnalyticsAgent.

    Extends Executor to support multiple retriever functions based on retriever types.
    """

    agent_class: str = "AnalyticsAgent"

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
        """Initialize the AnalyticsExecutor.

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
            ValueError: If an unsupported executor type is specified or required configuration is missing.
            Exception: For any other errors during initialization.
        """
        # Initialize func as a dictionary of callables.
        # Perform setup based on retriever type specified in utility_config.
        self.func = {}
        try:
            executor_config = utility_config.get("executor_config")
            if not executor_config:
                error_msg = "executor_config is missing in utility_config."
                logger.error(error_msg)
                raise ValueError(error_msg)

            executor = executor_config.get("type")
            if not executor:
                error_msg = "executor_config.type is missing in utility_config."
                logger.error(error_msg)
                raise ValueError(error_msg)

            if executor == "PostgresExecutor":
                if "PostgresExecutor" not in func:
                    db_config = executor_config.get("db_config")
                    if not db_config:
                        error_msg = "db_config is missing in executor_config."
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    self.func["PostgresExecutor"] = PostgresAPI(db_config).execute_query
                    logger.debug("Initialized PostgresExecutor.")
                else:
                    self.func["PostgresExecutor"] = func["PostgresExecutor"]
                    logger.info(
                        "PostgresExecutor already exists in func; using the user-defined one."
                    )
            elif executor == "PandasExecutor":
                if "PandasExecutor" not in func:
                    tables = executor_config.get("tables")
                    if not tables:
                        error_msg = "tables is missing in executor_config."
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    self.func["PandasExecutor"] = PandasAPI(tables).execute_query
                    logger.debug("Initialized PandasExecutor.")
                else:
                    self.func["PandasExecutor"] = func["PandasExecutor"]
                    logger.info(
                        "PandasExecutor already exists in func; using the user-defined one."
                    )
            else:
                error_msg = f"Unsupported executor type: {executor}."
                logger.error(error_msg)
                raise ValueError(error_msg)

        except Exception as e:
            logger.exception("Error occurred during AnalyticsExecutor initialization.")
            # Re-raise the exception to indicate failure during initialization
            raise

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
            error_msg = f"'__executor__' must be a string, got {type(executor)}."
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
