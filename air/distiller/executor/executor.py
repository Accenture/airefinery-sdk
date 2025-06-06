import asyncio
import concurrent.futures
import logging
import inspect
import functools
from typing import Callable, Optional, Union, Dict

from air import __version__

logger = logging.getLogger(__name__)


def is_async_callable(func):
    """
    Check if the given function is an asynchronous callable.

    Args:
        func (Callable): The function to check.

    Returns:
        bool: True if the function is asynchronous, False otherwise.
    """
    while True:
        if asyncio.iscoroutinefunction(func):
            return True
        if isinstance(func, functools.partial):
            func = func.func
        elif hasattr(func, "__wrapped__"):
            func = func.__getattribute__("__wrapped__")
        else:
            return False


# pylint: disable=too-many-instance-attributes
class Executor:
    """
    Executor class for handling the execution of synchronous or asynchronous functions.

    Attributes:
        agent_class (str): The name of the agent class.
    """

    agent_class: str = "CustomAgent"

    def __init__(
        self,
        func: Union[Callable, Dict[str, Callable]],
        send_queue: asyncio.Queue,
        account: str,
        project: str,
        uuid: str,
        role: str,
        return_string: bool = True,
        **_kwargs,  # Renamed to _kwargs to indicate it's intentionally unused
    ):
        # pylint: disable=too-many-positional-arguments
        self.func = func
        self.send_queue = send_queue
        self.account = account
        self.project = project
        self.uuid = uuid
        self.role = role
        self.return_string = return_string

        # Initialize the ThreadPoolExecutor
        self.executor = concurrent.futures.ThreadPoolExecutor()

    async def __call__(
        self,
        request_id: str,
        *args,
        func: Optional[Callable] = None,
        **kwargs,
    ):
        if func is None:
            assert isinstance(self.func, Callable)
            func = self.func

        # Execute the function in ThreadPoolExecutor regardless of sync or async.
        logger.debug("Executing function in ThreadPoolExecutor.")
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._run_function,
                func,
                args,
                kwargs,
            )
        except Exception as e:
            logger.exception("Exception in function execution")
            print("Exception in function execution")
            raise RuntimeError(f"Exception in function execution: {e}") from e

        # Validate the result
        result = self.validate_result(result)

        # Process the result
        res_content = str(result) if self.return_string else result

        # Send the result to send_queue
        await self.send_queue.put(
            {
                "account": self.account,
                "project": self.project,
                "uuid": self.uuid,
                "role": self.role,
                "request_args": {"content": res_content},
                "request_type": "executor",
                "request_id": request_id,
            }
        )

        logger.debug("Result sent to send_queue.")

        return result

    def validate_result(self, result):
        """
        Validates if the result is in a correct format.
        The executors that need result validation must
        have their own implementations
        """

        return result

    def _filter_arguments(self, func: Callable, args: tuple, kwargs: dict):
        """
        Filter the arguments to match the function signature.

        Args:
            func (Callable): The function to inspect.
            args (tuple): Positional arguments provided.
            kwargs (dict): Keyword arguments provided.

        Returns:
            tuple: Filtered positional arguments.
            dict: Filtered keyword arguments.
        """
        # Get the signature of the function
        sig = inspect.signature(func)
        params = sig.parameters

        # Check if the function accepts *args (variable positional arguments)
        accepts_varargs = any(
            param.kind == inspect.Parameter.VAR_POSITIONAL for param in params.values()
        )

        # Check if the function accepts **kwargs (variable keyword arguments)
        accepts_varkwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )

        # Determine which positional arguments are accepted
        positional_params = [
            name
            for name, param in params.items()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        # Limit positional arguments to what the function accepts
        if accepts_varargs:
            # If *args is accepted, pass all positional arguments
            filtered_args = args
        else:
            max_positional_args = len(positional_params)
            filtered_args = args[:max_positional_args]

        # Determine which keyword arguments are accepted
        if accepts_varkwargs:
            # If **kwargs is accepted, pass all keyword arguments
            filtered_kwargs = kwargs
        else:
            accepted_kwargs = {
                name
                for name, param in params.items()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }

            # Filter keyword arguments to include only accepted ones
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_kwargs}

        return filtered_args, filtered_kwargs

    def _run_function(self, func: Callable, args: tuple, kwargs: dict):
        # Execute the function and handle exceptions
        try:
            filtered_args, filtered_kwargs = self._filter_arguments(func, args, kwargs)
            if asyncio.iscoroutinefunction(func):
                # Create a new event loop in this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                # Run the async function in this loop
                result = loop.run_until_complete(
                    func(*filtered_args, **filtered_kwargs)
                )
                loop.close()
                return result
            else:
                return func(*filtered_args, **filtered_kwargs)
        except Exception as e:
            logger.exception("Exception in function execution")
            raise e

    def shutdown(self):
        """
        Shut down the thread pool executor.

        This should be called when the executor is no longer needed to release resources.
        """
        logger.debug("Shutting down executor.")
        self.executor.shutdown()
