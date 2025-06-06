import asyncio
import inspect
import json
from typing import Callable, Dict, Any, get_type_hints
import logging

from air.distiller.executor.executor import Executor

# Set up logging
logger = logging.getLogger(__name__)


class ToolExecutor(Executor):
    """Executor class for ToolUseAgent.

    Extends Executor to support multiple tool functions.
    """

    agent_class: str = "ToolUseAgent"

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
        """Initialize the ToolExecutor.

        Args:
            func (Dict[str, Callable]): A dictionary mapping function names to callables.
            send_queue (asyncio.Queue): Queue to send output to.
            account (str): Account identifier.
            project (str): Project identifier.
            uuid (str): User UUID.
            role (str): Role of the executor (typically the agent name).
            utility_config (Dict[str, Any]): Configuration dictionary for utility agents.
            return_string (bool): Whether to return a stringified output back.

        Raises:
            ValueError: If an unsupported function name is specified or required configuration is missing.
            Exception: For any other errors during initialization.
        """
        logger.debug(
            f"Initializing ToolExecutor with role='{role}', account='{account}', project='{project}', uuid='{uuid}'"
        )

        # Initialize func as a dictionary of callables.
        # Perform setup based on function names specified in utility_config.
        self.func = {}
        try:
            custom_tools = utility_config.get("custom_tools", [])
            if not custom_tools:
                logger.warning("No custom tools specified in utility_config.")

            for idx, tool_json in enumerate(custom_tools):
                try:
                    tool_dict = json.loads(tool_json)
                    function_name = tool_dict["function"]["name"]

                    if function_name not in func:
                        error_msg = (
                            f"Function '{function_name}' specified in utility_config is not "
                            f"provided in 'func' dictionary."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    self.func[function_name] = func[function_name]
                    logger.debug(f"Added function '{function_name}' to ToolExecutor.")

                except json.JSONDecodeError as jde:
                    error_msg = (
                        f"Invalid JSON in custom_tools at index {idx}: {tool_json}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg) from jde
                except KeyError as ke:
                    error_msg = f"Missing key {ke} in tool definition at index {idx}: {tool_json}"
                    logger.error(error_msg)
                    raise ValueError(error_msg) from ke
        except Exception as e:
            logger.exception("Error occurred during ToolExecutor initialization.")
            # Re-raise the exception to indicate failure during initialization
            raise

        # Initialize the base class with the func dictionary
        try:
            super().__init__(
                func=self.func,
                send_queue=send_queue,
                account=account,
                project=project,
                uuid=uuid,
                role=role,
                return_string=return_string,
            )
            logger.debug(
                f"ToolExecutor initialized successfully with {len(self.func)} functions."
            )
        except Exception as e:
            logger.exception(
                "Error occurred during ToolExecutor superclass initialization."
            )
            raise

    async def __call__(self, request_id: str, *args, **kwargs):
        """Execute the appropriate tool function based on __executor__.

        Args:
            request_id (str): Unique identifier for the request.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of executing the selected tool function.

        Raises:
            ValueError: If '__executor__' is not specified or invalid.
            TypeError: If '__executor__' is not a string.
        """
        executor = kwargs.pop("__executor__", None)
        if executor is None:
            error_msg = f"'__executor__' must be specified in kwargs for request_id '{request_id}'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(executor, str):
            error_msg = f"'__executor__' must be a string, got {type(executor)} for request_id '{request_id}'."
            logger.error(error_msg)
            raise TypeError(error_msg)

        if executor not in self.func:
            error_msg = f"Tool '{executor}' is not available in func for request_id '{request_id}'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        selected_func = self.func[executor]
        logger.debug(
            f"Executing tool function '{executor}' for request_id '{request_id}'."
        )

        # -----------------------------------------------------
        # Automatically convert kwargs from str to int/float
        # based on selected_func’s signature and type hints.
        # -----------------------------------------------------
        signature = inspect.signature(selected_func)
        type_hints = get_type_hints(selected_func)
        for param_name, param in signature.parameters.items():
            if param_name in kwargs:
                expected_type = type_hints.get(param_name)
                current_value = kwargs[param_name]

                # Only convert if the current value is a string
                # and the type hint is int or float
                if isinstance(current_value, str) and expected_type in (int, float):
                    try:
                        if expected_type is int:
                            kwargs[param_name] = int(current_value)
                        elif expected_type is float:
                            kwargs[param_name] = float(current_value)
                    except ValueError:
                        logger.warning(
                            f"Failed to convert '{param_name}' from string to {expected_type}; "
                            f"original value='{current_value}'. Leaving as string."
                        )

        try:
            # Call the base class __call__ method with the selected function
            result = await super().__call__(
                request_id=request_id, func=selected_func, *args, **kwargs
            )
            logger.debug(
                f"Successfully executed tool function '{executor}' for request_id '{request_id}'."
            )
            return result
        except Exception as e:
            logger.exception(
                f"Error executing tool function '{executor}' for request_id '{request_id}'."
            )
            # Re-raise the exception to allow upstream handling
            raise
