import asyncio
import json
import re
import traceback
from importlib.metadata import version
from typing import Any, Callable, Optional, TypeVar, Union, cast

import requests
import websockets
from omegaconf import OmegaConf
from pydantic import BaseModel, ValidationError

from air import BASE_URL, PostgresAPI, __version__
from air.auth.token_provider import TokenProvider
from air.distiller.executor import (
    get_all_executor_agents,
    get_executor,
)
from air.distiller.pii_handler.pii_handler import PIIHandler
from air.types.base import CustomBaseModel
from air.types.distiller.client import (
    DistillerIncomingMessage,
    DistillerMemoryOutgoingMessage,
    DistillerMemoryRequestArgs,
    DistillerMemoryRequestType,
    DistillerMessageRequestArgs,
    DistillerMessageRequestType,
    DistillerOutgoingMessage,
    DistillerPongMessage,
)
from air.utils import async_input, async_print, get_base_headers, get_base_headers_async

logger = __import__("logging").getLogger(__name__)


def string_check(s) -> None:
    """
    Validate that the input string contains only letters, numbers, hyphens,
    and underscores.

    Parameters:
    s (str): The string to validate.

    Raises:
    ValueError: If the string contains invalid characters.
    """
    # Define the regex pattern to match only allowed characters:
    # alphabets, numbers, hyphens, and underscores
    pattern = re.compile(r"^[a-zA-Z0-9_-]+$")

    # Use the fullmatch method to check if the entire string matches the pattern
    if pattern.fullmatch(s):
        return

    # Raise ValueError if the string does not match the allowed characters
    raise ValueError(
        f"Invalid string '{s}'. The string can only contain alphabets, numbers, "
        f"hyphens ('-'), and underscores ('_')."
    )


RequestArgsType = TypeVar("RequestArgsType", bound=CustomBaseModel)


def _build_request_args(
    ModelClass: type[RequestArgsType], kwargs: dict
) -> RequestArgsType:
    """
    Validate kwargs and build a CustomBaseModel instance.

    Args:
        ModelClass: A subclass of pydantic.BaseModel.
        kwargs: Keyword arguments to validate.

    Returns:
        An instance of ModelClass.

    Raises:
        ValidationError: If required fields are missing or invalid.
    """
    expected_fields = set(ModelClass.model_fields.keys())
    extra_fields = set(kwargs.keys()) - expected_fields
    if extra_fields:
        logger.warning(
            f"Unexpected fields: {extra_fields}. Expected fields: {expected_fields}"
        )
    try:
        return ModelClass.model_validate(kwargs)
    except ValidationError as e:
        logger.error(
            f"Validation error for {ModelClass.__name__}: {e}. Expected fields: {expected_fields}"
        )
        raise


class AsyncDistillerClient:
    """
    Distiller SDK for AI Refinery.

    This class provides an interface for interacting with the AI Refinery's
    distiller service, allowing users to create projects, download configurations,
    and run distiller sessions.
    """

    # Define API endpoints for various operations
    run_suffix = "distiller/run"
    create_suffix = "distiller/create"
    download_suffix = "distiller/download"
    reset_suffix = "distiller/reset"
    max_size_ws_recv = 167772160
    ping_interval = 10

    def __init__(
        self, api_key: str | TokenProvider, *, base_url: str = BASE_URL, **kwargs
    ) -> None:
        """
        Initialize the AsyncDistillerClient with authentication details.

        Args:
            api_key (str | TokenProvider): Credential that will be placed in the
                ``Authorization`` header of every request.
                * **str** – a raw bearer token / API key.
               * **TokenProvider** – an object whose ``token()`` (and
                  ``token_async()``) method returns a valid bearer token.  The
                  client calls the provider automatically before each request.

            base_url (str, optional): Base URL for the API. Defaults to "".
        """
        super().__init__()

        # Use the provided base URL or the default one
        self.base_url = base_url
        self.api_key = api_key

        self.account = self.validate_api_key(api_key)
        assert self.account, "Authentication failed."

        # Initialize other attributes
        self.project = None
        self.uuid = None
        self.connection = None
        self.executor_dict = None

        # Initialize background tasks
        self._ping_task = None
        self._send_task = None
        self._receive_task = None

        # Initialize message queues
        self.send_queue = None
        self.receive_queue = None

        # Initialize last ping timestamp
        self._last_ping_received = None

        # Initialize background tasks tracker
        self._wait_task_list = None

        # PII Handler potion (useful to identify & mask/unmask sensitive information)
        self.pii_handler = None

    def validate_api_key(self, api_key: str | TokenProvider):
        """
        Sends a POST request to validate the given API key.

        Parameters:
        api_key (str): The API key to be validated.

        Returns:
        response (requests.Response): The response object from the server.
        """
        headers = get_base_headers(api_key)

        try:
            response = requests.post(
                f"{self.base_url}/authentication/validate", headers=headers
            )
            response.raise_for_status()  # Raise an error for bad responses
            response_json = response.json()
            organization_id = response_json.get("organization_id")

            return organization_id
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    def create_project(
        self,
        *,
        project: str,
        config_path: Optional[str] = None,
        json_config: Optional[dict] = None,
    ) -> bool:
        """
        Create a project based on the configuration file specified by the config path. (REST API)

        Args:
            config_path (str): Path to the configuration file.
            json_config (str): json version of the yaml config
            project (str): Name of the project to be created.

        Returns:
            bool: True if the project is successfully created, False otherwise.
        """
        print(f"Registering project '{project}' for account '{self.account}'")
        string_check(project)

        if config_path:
            # Load the YAML configuration file
            yaml_config = OmegaConf.load(config_path)
            # Resolve the YAML config into a JSON format
            json_config = cast(dict, OmegaConf.to_container(yaml_config, resolve=True))

        if not json_config:
            raise Exception("Either json_config or config_path must be provided.")

        # Prepare the payload for the request
        payload = {
            "project": project,
            "config": json_config,
            "sdk_version": __version__,
        }

        # Prepare the headers with the API key for authentication
        headers = get_base_headers(
            self.api_key,
            extra_headers={
                "airefinery_account": str(self.account),
            },
        )

        # Determine the base URL
        base_url = f"{self.base_url}/{self.create_suffix}"

        # Send a POST request to create the project
        response = requests.post(base_url, headers=headers, data=json.dumps(payload))
        # Check the response status and return the result
        if response.status_code == 201:
            try:
                repsonse_content = json.loads(response.content)
                print(
                    f"Project {project} - version {repsonse_content['project_version']} "
                    f"has been created for {self.account}."
                )
            except json.JSONDecodeError:
                print(f"Project {project} has been created for {self.account}.")
            return True
        else:
            print("Failed to create the project.")
            print(response)
            return False

    def download_project(
        self,
        project: str,
        project_version: Optional[str] = None,
        sdk_version: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Download the configuration from the server for a given project. (REST API)

        Args:
            project (str): Name of the project to download the configuration for.
            project_version: Optional(str): Version number of the project to download.
        Returns:
            dict: The downloaded configuration as a JSON object, or None if the download fails.
        """
        string_check(project)

        # Prepare the payload for the request
        payload = {
            "project": project,
            "project_version": project_version,
            "sdk_version": __version__,
        }

        # Prepare the headers with the API key for authentication
        headers = get_base_headers(
            self.api_key,
            extra_headers={
                "airefinery_account": str(self.account),
            },
        )

        # Determine the base URL
        base_url = f"{self.base_url}/{self.download_suffix}"

        # Send a POST request to download the project configuration
        response = requests.post(base_url, headers=headers, data=json.dumps(payload))

        # Return the JSON configuration if the request is successful
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            print("Failed to download the config")

    async def _send_loop(self):
        """
        The send loop task will run forever until the task is cancled. It will
        get the message from the queue and send it back to the server.
        """
        try:
            assert self.send_queue is not None
            assert self.connection is not None
            while True:
                message = await self.send_queue.get()
                if message is None:
                    break  # Exit the loop if None is received
                await self.connection.send(message.model_dump_json())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Send loop error: {e}")

    async def _receive_loop(self):
        """
        The receive loop task will run forever until the task is cancled. It will
        get the message from the server and add it to the receive queue
        """
        try:
            assert self.connection is not None
            assert self.receive_queue
            while True:
                message = await self.connection.recv()
                try:
                    msg = json.loads(message)
                except:
                    print(f"Receive non json object {message}")
                    continue

                if msg.get("type", None) == "PING":
                    await self.send(DistillerPongMessage())
                    self._last_ping_received = asyncio.get_event_loop().time()
                    continue

                msg = DistillerIncomingMessage.model_validate(msg)
                msg = await asyncio.to_thread(self.unmask_pii_if_needed, msg)

                await self.receive_queue.put(msg)

        except asyncio.CancelledError:
            pass

        except Exception as e:
            print(f"Receive loop error: {e}")

    async def _ping_monitor(self):
        """
        The ping monitor task will run forever until the task is cancled. It will
        check if the client has recieved a heart beat from the server. If not,
        it will close the websocket connection and break.
        """
        assert self._last_ping_received, "last ping received cannot be None"
        try:
            while True:
                await asyncio.sleep(self.ping_interval)
                now = asyncio.get_event_loop().time()
                if now - self._last_ping_received > 2 * self.ping_interval:
                    print(
                        "Ping monitor: No PING received in the last interval. Closing connection."
                    )
                    await self.close()
                    break
        except asyncio.CancelledError:
            pass

    def mask_payload_if_needed(
        self, payload: DistillerOutgoingMessage
    ) -> DistillerOutgoingMessage:
        """
        Mask PII in payload if protection is enabled and payload contains a query.

        Args:
            payload: The payload to potentially mask

        Returns:
            The payload with PII masked if applicable
        """

        if not isinstance(payload.request_args, DistillerMessageRequestArgs):
            return payload

        if (
            self.pii_handler
            and self.pii_handler.is_enabled()
            and payload.request_type == DistillerMessageRequestType.QUERY
            and payload.request_args.query
        ):
            original = payload.request_args.query
            masked_query, metadata = self.pii_handler.mask_text(original)
            payload["request_args"]["query"] = masked_query

            if metadata:
                self.pii_handler.extend_metadata(metadata)

        return payload

    def unmask_pii_if_needed(
        self, msg: DistillerIncomingMessage
    ) -> DistillerIncomingMessage:
        """
        Unmask PII in message content if PII handler is enabled and content exists.

        Args:
            msg: The message dictionary to potentially unmask

        Returns:
            The message with PII unmasked if applicable
        """
        if self.pii_handler and self.pii_handler.is_enabled() and msg.content:
            original_masked = msg.content
            demasked = self.pii_handler.demask_text(
                original_masked,
                self.pii_handler.get_metadata(),
            )
            msg.content = demasked

        return msg

    async def send(
        self,
        payload: (
            DistillerOutgoingMessage
            | DistillerMemoryOutgoingMessage
            | DistillerPongMessage
        ),
    ) -> None:
        """
        Enqueue a payload to be sent over the established websocket connection.

        Args:
        payload (dict): The payload to send.
        """
        # Apply PII masking if needed
        if isinstance(payload, DistillerOutgoingMessage):
            masked_payload = self.mask_payload_if_needed(payload)
        else:
            masked_payload = payload

        assert self.send_queue
        await self.send_queue.put(masked_payload)

    async def recv(self) -> DistillerIncomingMessage:
        """
        Dequeue a message from the receive queue.
        """
        while True:
            if self.receive_queue is None:
                raise ConnectionError("Receive queue is empty after disconnect.")

            try:
                msg = await asyncio.wait_for(self.receive_queue.get(), 0.1)
            except TimeoutError:
                continue
            return msg

    async def connect(
        self,
        project: str,
        uuid: str,
        project_version: Optional[str] = None,
        custom_agent_gallery: Optional[dict[str, Callable | dict]] = None,
        executor_dict: Optional[dict[str, Callable | dict]] = None,
    ) -> None:
        """
        Connect to the account/project/uuid-specific URL.

        Args:
            project (str): Name of the project.
            uuid (str): Unique identifier for the session.
            custom_agent_gallery (Optional[dict[str, Callable]], optional):
                        Custom agent handlers. Defaults to None.
        """
        string_check(project)
        string_check(uuid)

        headers = await get_base_headers_async(
            self.api_key,
            extra_headers={
                "airefinery_account": str(self.account),
            },
        )

        if project_version:
            # Directly load the versioned project
            self.project = f"{project}:{project_version}"
        else:
            # Load the latest project on the fly
            self.project = project
        self.uuid = uuid

        # Determine the base URL
        base_url = f"{self.base_url}/{self.run_suffix}"

        # Establish a websocket connection between the client and the server
        base_url = base_url.replace("http", "ws")
        base_url = base_url.replace("https", "wss")

        try:
            # Compare the version
            if version("websockets") >= "14.0":
                self.connection = await websockets.connect(
                    f"{base_url}/{self.account}/{self.project}/{uuid}",
                    additional_headers=headers,
                    max_size=self.max_size_ws_recv,
                )
            else:
                self.connection = await websockets.connect(
                    f"{base_url}/{self.account}/{self.project}/{uuid}",
                    extra_headers=headers,
                    max_size=self.max_size_ws_recv,
                )

            # Start background tasks after successful connection
            self._send_task = asyncio.create_task(self._send_loop())
            self._receive_task = asyncio.create_task(self._receive_loop())

            # Start the ping monitor task
            self._ping_task = asyncio.create_task(self._ping_monitor())
            self._last_ping_received = asyncio.get_event_loop().time()

            self.send_queue = asyncio.Queue()
            self.receive_queue = asyncio.Queue()

            self._wait_task_list = []

            if custom_agent_gallery is None:
                custom_agent_gallery = {}
            if executor_dict is None:
                executor_dict = {}

            if len(custom_agent_gallery) > 0:
                executor_dict = custom_agent_gallery
                print(
                    "The custom_agent_gallery argument is going to be deprecated "
                    "in future release. Please use executor_dict in the future."
                )

            # Load the latest project config for the user
            project_config_dict = self.download_project(
                project, project_version, __version__
            )
            if not project_config_dict:
                raise ValueError("Project configuration could not be loaded.")

            project_config = json.loads(json.loads(project_config_dict["config"]))
            self.initialize_executor(
                project=project,
                project_config=project_config,
                project_version=project_version,
                executor_dict=executor_dict,
            )
            base_config = project_config.get("base_config", {})
            pii_config = base_config.get("pii_masking", {})
            if pii_config.get("enable", False):
                self.pii_handler = PIIHandler()
                self.pii_handler.enable()
                self.pii_handler.load_runtime_overrides(project_config)

        except Exception as e:
            print(f"Failed to connect: {e}")
            self.connection = None
            raise

    def initialize_executor(
        self,
        project: str,
        project_config: Any,
        project_version: Optional[str] = None,
        executor_dict: Optional[dict[str, Callable | dict[str, Callable]]] = None,
    ):
        """Initialize the executor based on the project config and the provided executor_dict."""
        # Default executor_dict to an empty dictionary if not provided
        if executor_dict is None:
            executor_dict = {}

        # Reset the executors
        self.executor_dict = {}

        # Walk through each utility config to ensure all the executors are properly initialized
        for u_cfg in project_config.get("utility_agents", []):
            agent_name = u_cfg["agent_name"]
            agent_class = u_cfg["agent_class"]

            # Share all tools with the tool use agent (backward compatibility)
            if agent_class == "ToolUseAgent":
                # Determine the appropriate executor_dict entry for the agent
                agent_executor = executor_dict.get(agent_name)

                if isinstance(agent_executor, Callable) or (not agent_executor):
                    # If agent_executor is a Callable (or doesn't exist),
                    # replace with a dictionary of all Callables
                    executor_dict[agent_name] = {
                        name: func
                        for name, func in executor_dict.items()
                        if isinstance(func, Callable)
                    }
                elif isinstance(agent_executor, dict):
                    # If agent_executor is already a dict, leave it unchanged
                    pass

            if agent_class in get_all_executor_agents():
                # This agent requires an executor
                # Create the executor wrapper for the agent
                self.executor_dict[agent_name] = get_executor(
                    agent_class=agent_class,
                    func=executor_dict.get(agent_name, {}),
                    send_queue=self.send_queue,
                    account=self.account,
                    project=self.project,
                    uuid=self.uuid,
                    role=agent_name,
                    utility_config=u_cfg.get("config", {}),
                )

    async def close(self) -> None:
        """
        Close the websocket connection and cancel background tasks.
        """
        if self.pii_handler:
            self.pii_handler.clear_mapping()
            self.pii_handler.clear_metadata()
        tasks = [self._send_task, self._receive_task, self._ping_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close the send queue
        if self.send_queue:
            if not self.send_queue.empty():
                await self.send_queue.put(None)
            self.send_queue = None

        # Close Wait message related tasks
        if self._wait_task_list:
            for task in self._wait_task_list:
                if task:
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=1)
                    except TimeoutError:
                        print(".")
                    except asyncio.CancelledError:
                        print("cancellation error")

        # Close the connection
        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                print(f"Failed to close connection: {e}")
            finally:
                self.connection = None
                self.project = None
                self.uuid = None
                self.executors = None
                self._wait_task_list = None
                self.receive_queue = None

    async def request(
        self,
        request_type: DistillerMemoryRequestType | DistillerMessageRequestType,
        request_args: (
            Union[DistillerMessageRequestArgs, DistillerMemoryRequestArgs] | None
        ),
        **kwargs,
    ):
        """
        Submit a request to the websocket.

        Args:
            request_type (DistillerMemoryRequestType or DistillerMessageRequestType): Type of the request.
            request_args (dict): Arguments for the request.
            **kwargs: Additional keyword arguments.
        """
        assert self.project, "Project cannot be None. You should call connect first."
        assert self.uuid, "uuid cannot be None. You should call connect first."

        message_kwargs = {
            "project": self.project,
            "uuid": self.uuid,
            "request_args": request_args,
            "request_type": request_type,
            "role": "user",
        }

        try:
            outgoing_message_class, outgoing_message_args = None, None

            if request_type in DistillerMemoryRequestType:
                outgoing_message_class = DistillerMemoryOutgoingMessage
                outgoing_message_args = DistillerMemoryRequestArgs
            elif request_type in DistillerMessageRequestType:
                outgoing_message_class = DistillerOutgoingMessage
                outgoing_message_args = DistillerMessageRequestArgs

            if not outgoing_message_class or not outgoing_message_args:
                raise ValueError(
                    f"Invalid message_types: {request_type}. or message_args: {request_args}. "
                    f"Allowed types are: {tuple(item.value for item in DistillerMessageRequestType)} and "
                    f"{tuple(item.value for item in DistillerMemoryRequestType)}"
                )

            outgoing_message_args.model_validate(request_args)
            payload = outgoing_message_class(**message_kwargs)

        except Exception as e:
            raise ValueError("Failed to validate the request.") from e

        await self.send(payload)

        db_client = kwargs.get("db_client", None)

        try:
            while True:
                try:
                    msg = await self.recv()
                except ConnectionError:
                    return
                msg = DistillerIncomingMessage.model_validate(msg)
                status = msg.status
                role = msg.role

                if status == "complete":
                    break

                elif status == "wait":
                    assert self._wait_task_list is not None

                    assert self.executor_dict, "executor_dict cannot be None"
                    assert role in self.executor_dict, (
                        f"Cannot find {role} from the executor_dict: "
                        f"{self.executor_dict.keys()}"
                    )

                    assert self.send_queue

                    wait_msg_task = asyncio.create_task(
                        self.executor_dict[role](
                            request_id=msg.request_id, **msg.kwargs
                        )
                    )
                    self._wait_task_list.append(wait_msg_task)

                if status not in ["wait", "complete"]:
                    if msg.content and db_client:
                        await self._log_chat(
                            db_client=db_client,
                            project=self.project,
                            uuid=self.uuid,
                            message=msg,
                        )
                    if role != "user":
                        yield msg
        except websockets.ConnectionClosedOK:
            print("Connection closed gracefully by the server")
        except websockets.ConnectionClosedError as e:
            print(f"Connection closed with error: {e}")
        except Exception as e:
            print(traceback.format_exc())
            raise e

    async def query(self, **kwargs):
        """
        Send a query request to the websocket, with PII masked if enabled.

        Args:
            query (str): The query string.
            image (Optional[str], optional): Image to include in the query. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Coroutine: The request coroutine.
        """

        request_args = _build_request_args(DistillerMessageRequestArgs, kwargs)

        return self.request(
            request_type=DistillerMessageRequestType.QUERY,
            request_args=request_args,
            **kwargs,
        )

    async def retrieve_memory(self, **kwargs):
        """
        Retrieve memory from the websocket.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            The retrieved memory.
        """

        request_args = _build_request_args(DistillerMemoryRequestArgs, kwargs)
        responses = self.request(
            request_type=DistillerMemoryRequestType.RETRIEVE,
            request_args=request_args,
            **kwargs,
        )
        content = ""

        async for response in responses:
            if response.role == "memory":
                content = response.content
        return content

    async def add_memory(self, **kwargs):
        """
        Add memory to the websocket.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """

        request_args = _build_request_args(DistillerMemoryRequestArgs, kwargs)

        responses = self.request(
            request_type=DistillerMemoryRequestType.ADD,
            request_args=request_args,
        )
        async for _ in responses:
            pass

    async def reset_memory(self):
        """
        Reset memory in the websocket.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            Coroutine: The request coroutine.
        """
        responses = self.request(
            request_type=DistillerMemoryRequestType.RESET,
            request_args=DistillerMemoryRequestArgs(),
        )
        async for _ in responses:
            pass

    async def retrieve_history(
        self,
        *,
        db_client: PostgresAPI,
        project: str,
        uuid: str,
        n_messages: int,
        as_string=False,
    ) -> str | list[dict]:
        """
        Retrieve chat history from the database.

        Args:
            db_client (PostgresAPI): Database client.
            project (str): Name of the project.
            uuid (str): Unique identifier for the session.
            n_messages (int): Number of messages to retrieve.
            as_string (bool, optional): Whether to return the history as a string.
                                        Defaults to False.

        Returns:
            str | list[dict]: Chat history as a string or list of dictionaries.
        """
        table_name = f"public.backend_information_{self.account}_{project}"
        query = f"""SELECT full_content
                    FROM {table_name}
                    WHERE uuid = %s
                    ORDER BY timestamp DESC
                    LIMIT %s;"""
        response, success = await db_client.execute_query(query, [uuid, n_messages])
        if not success or not response:
            print(
                f"Failed to retrieve past history for {self.account}_{project}_{uuid}."
            )
            return "" if as_string else []

        messages = []
        for msg_str in response:
            try:
                msg = json.loads(
                    msg_str[0]
                )  # Ensure the content is a valid JSON structure
                messages.append(msg)
            except json.JSONDecodeError as e:
                print(msg_str)
                raise e

        if as_string:
            out = ""
            for msg in reversed(messages):
                out += f"JSONSTART{json.dumps(msg)}JSONEND"
            return out
        else:
            return messages

    async def _log_chat(
        self,
        *,
        db_client: PostgresAPI,
        project: str,
        uuid: str,
        message: DistillerIncomingMessage,
    ) -> bool:
        """
        Log conversation history to a database.

        Each account + project will get its own table named
        backend_information_accountname_project_name.

        Table Schema:
        - uuid_timestamp VARCHAR: unique user ID + timestamp to trace chat response messages
        - uuid VARCHAR: user ID
        - timestamp FLOAT: unix timestamp
        - role TEXT: agent in use
        - content TEXT: agent response content
        - full_content TEXT: full communication message from the distiller service.

        Args:
            db_client (PostgresAPI): Database client.
            project (str): Name of the project.
            uuid (str): Unique identifier for the session.
            message (dict): Message to log.

        Returns:
            bool: True if logging is successful, False otherwise.
        """
        assert self.account
        account = self.account.replace("-", "_")
        table_name = f"public.backend_information_{account}_{project}"
        print(f"TABLE NAME: {table_name}")

        table_creation_query = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                    uuid_timestamp VARCHAR,
                    uuid VARCHAR,
                    timestamp FLOAT,
                    role TEXT,
                    content TEXT,
                    full_content TEXT
                );"""
        _, creation_response_success = await db_client.execute_query(
            table_creation_query
        )
        if not creation_response_success:
            print(
                "Failed to create the account project table to log chat history in the database."
            )
            return False

        insert_query = f"""INSERT INTO {table_name} (uuid_timestamp, uuid, timestamp, role, content, full_content)
                           VALUES (%s, %s, %s, %s, %s, %s);"""
        _, insertion_response_success = await db_client.execute_query(
            insert_query,
            params=[
                message.uuid_timestamp,
                uuid,
                message.timestamp,
                message.role,
                message.content,
                message.model_dump_json(),
            ],
        )
        if not insertion_response_success:
            print("Failed to upload the json output to the database.")
            return False
        return True

    def __call__(self, **kwargs) -> "_DistillerContextManager":
        """
        Return a context manager for connecting to the Distiller server.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            _DistillerContextManager: The context manager instance.
        """
        return self._DistillerContextManager(self, **kwargs)

    class _DistillerContextManager:
        def __init__(self, client: "AsyncDistillerClient", **kwargs):
            self.client = client
            self.kwargs = kwargs

        async def __aenter__(self):
            await self.client.connect(**self.kwargs)
            return self.client

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.client.close()

    async def _interactive_helper(
        self,
        project: str,
        uuid: str,
        executor_dict: Optional[dict[str, Callable | dict[str, Callable]]] = None,
    ):
        """
        Helper function for interactive mode.

        Args:
            project (str): Name of the project.
            uuid (str):    Unique identifier for the session.
            executor_dict (dict[str, Callable], optional):
                           Custom agent handlers. Defaults to {}.
        """
        async with self(
            project=project,
            uuid=uuid,
            executor_dict=executor_dict,
        ) as dc:
            while True:
                query = await async_input("%%% USER %%%\n")
                responses = await dc.query(query=query)
                async for response in responses:
                    if (not response.role) or (not response.content):

                        continue

                    await async_print()
                    await async_print(f"%%% AGENT {response.role} %%%")
                    await async_print(response.content)
                    await async_print()
                    pass

    def interactive(
        self,
        project: str,
        uuid: str,
        custom_agent_gallery: Optional[
            dict[str, Callable | dict[str, Callable]]
        ] = None,
        executor_dict: Optional[dict[str, Callable | dict[str, Callable]]] = None,
    ):
        """
        Enter interactive mode, allowing the user to interact with the agents through the terminal.

        Args:
            project (str): Name of the project.
            uuid (str):    Unique identifier for the session.
            custom_agent_gallery (dict[str, Callable], optional):
                           Custom agent handlers. Defaults to {}.
        """
        # Get the event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:  # No event loop is running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if custom_agent_gallery is not None:
            executor_dict = custom_agent_gallery
            print(
                "The argument custom_agent_gallery is going to be "
                "deprecated in future releases. "
                "Please use executor_dict in the future."
            )

        # Run the asynchronous function using the event loop
        loop.run_until_complete(self._interactive_helper(project, uuid, executor_dict))
