import asyncio
import inspect
import json
import logging
from typing import Any, Callable, Dict, get_type_hints

from air.distiller.executor.executor import Executor, is_async_callable
from air.types.distiller.client import (
    DistillerMessageRequestArgs,
    DistillerMessageRequestType,
    DistillerOutgoingMessage,
)
from air.types.distiller.executor.tool_use_config import ToolUseConfig

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
        tool_use_config = ToolUseConfig(**utility_config)
        self._wait_time = tool_use_config.wait_time
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

        loop = asyncio.get_running_loop()
        try:
            if is_async_callable(selected_func):
                filtered_args, filtered_kwargs = self._filter_arguments(
                    selected_func, args, kwargs
                )
                exec_task = selected_func(*filtered_args, **filtered_kwargs)
            else:
                exec_task = loop.run_in_executor(
                    self.executor,
                    self._run_function,
                    selected_func,
                    args,
                    kwargs,
                )
            result = await asyncio.wait_for(exec_task, timeout=self._wait_time)
            result = self.validate_result(result)
            logger.debug(
                "Successfully executed tool function '%s' for request_id '%s'.",
                executor,
                request_id,
            )
        except asyncio.TimeoutError:
            result = json.dumps(
                {
                    "error": {
                        "type": "timeout",
                        "message": "Tool execution timed out.",
                        "timeout_sec": self._wait_time,
                        "request_id": request_id,
                    }
                }
            )
            logger.info(
                "Tool function '%s' timed out for request_id '%s'.",
                executor,
                request_id,
            )
        except Exception as e:
            logger.exception(
                "Error executing tool function '%s' for request_id '%s'.",
                executor,
                request_id,
            )
            raise RuntimeError(f"Exception in tool function execution: {e}") from e

        res_content = (
            str(result) if (result is not None and self.return_string) else result
        )
        request_args = DistillerMessageRequestArgs(content=res_content)
        request = DistillerOutgoingMessage(
            account=self.account,
            project=self.project,
            uuid=self.uuid,
            role=self.role,
            request_args=request_args,
            request_type=DistillerMessageRequestType.EXECUTOR,
            request_id=request_id,
        )
        await self.send_queue.put(request)
        return result
