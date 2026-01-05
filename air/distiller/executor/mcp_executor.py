import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Literal, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from air.distiller.executor.executor import Executor
from air.types.distiller.client import (
    DistillerMessageRequestArgs,
    DistillerMessageRequestType,
    DistillerOutgoingMessage,
)
from air.types.distiller.executor.mcp_config import (
    MCPClientAgentConfig,
    MCPServerConfig,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _session_context(
    url: str,
    server_type: Literal["http-stream", "sse"] = "http-stream",
    operation_hint: str = "",
    headers: Optional[Dict[str, str]] = None,
):
    """
    Create a short‑lived MCP ClientSession bound to a single coroutine.
    Supports both SSE and streamable HTTP endpoints.

    Args:
        url: The MCP server endpoint URL
        server_type: The type of MCP server ('http-stream' or 'sse')
        operation_hint: Optional hint about the operation being performed
        headers: Optional HTTP headers to send with requests
    """
    # Prepare headers if provided
    extra_args = {}
    if headers:
        extra_args["headers"] = headers

    match server_type:
        case "http-stream":
            # Use streamablehttp_client for HTTP endpoints
            async with streamablehttp_client(url, **extra_args) as (
                read,
                write,
                *optional,
            ):
                # Store session_id getter for later use
                session_id_getter = None
                if optional and callable(optional[0]):
                    session_id_getter = optional[0]

                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Try to get session ID after initialization
                    if session_id_getter:
                        try:
                            session_id = session_id_getter()
                            hint = f" [{operation_hint}]" if operation_hint else ""
                            logger.info(
                                f"MCP session established with ID:{hint}: {session_id}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Non-critical: session ID unavailable after initialization. {e}"
                            )

                    yield session
        case "sse":
            # Use sse_client for SSE endpoints
            async with sse_client(url, **extra_args) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session


class MCPExecutor(Executor):
    """
    Executor for MCPClientAgent.
    """

    agent_class: str = "MCPClientAgent"

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        func: Dict[str, Callable],
        send_queue: asyncio.Queue,
        account: str,
        project: str,
        uuid: str,
        role: str,
        utility_config: Dict[str, Any],
        return_string: bool = True,
    ) -> None:

        mcp_config = MCPClientAgentConfig(**utility_config)

        # mcpServers is always populated (even for legacy configs)
        # because the validator converts single-server to multi-server format
        # Headers are already processed by MCPServerConfig validator
        self._servers: Dict[str, MCPServerConfig] = mcp_config.mcpServers

        # Locks to serialize access to each MCP server to prevent concurrency issues
        self._server_locks: Dict[str, asyncio.Lock] = {
            server_name: asyncio.Lock() for server_name in self._servers
        }

        # Mapping of tool names to server names (populated during tool discovery)
        self._tool_to_server: Dict[str, str] = {}

        super().__init__(
            func={},
            send_queue=send_queue,
            account=account,
            project=project,
            uuid=uuid,
            role=role,
            return_string=return_string,
        )

        servers_info = ", ".join(
            f"{name}: {config.url}"
            for name, config in self._servers.items()  # pylint: disable=no-member
        )
        logger.info("MCPExecutor initialized with servers: %s", servers_info)

    async def _list_tools_from_server(
        self, server_name: str, server_config: MCPServerConfig
    ) -> List[Dict[str, Any]]:
        """
        List tools from a single server.

        Args:
            server_name: Name of the server
            server_config: Configuration for the server

        Returns:
            List of tools in OpenAI format with server_url
        """
        server_url = server_config.url
        tools: List[Dict[str, Any]] = []

        async with self._server_locks[server_name]:
            try:
                async with _session_context(
                    server_config.url,
                    server_config.type,
                    f"listing tools from {server_name}",
                    headers=server_config.headers,
                ) as session:
                    tools_response = await session.list_tools()

                    for tool in tools_response.tools:
                        params_raw = tool.inputSchema or {}
                        if not isinstance(params_raw, dict):
                            logger.warning(
                                "Tool %s inputSchema isn't dict – coercing to object",
                                tool.name,
                            )
                            params_raw = {}

                        # Normalise to JSON‑Schema object form
                        if "type" not in params_raw:
                            params_raw = {
                                "type": "object",
                                "properties": params_raw,
                            }
                        params_raw.setdefault("properties", {})
                        params_raw.setdefault(
                            "required", list(params_raw["properties"].keys())
                        )

                        # Build OpenAI format with server URL metadata
                        format_tool = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": params_raw,
                            },
                            "server_url": server_url,
                        }
                        tools.append(format_tool)

                    logger.info(
                        "Successfully loaded %d tools from server '%s' (URL: %s)",
                        len(tools_response.tools),
                        server_name,
                        server_url,
                    )

            except Exception as e:
                # Do not raise error - allow partial failures
                # Some MCP servers may be unavailable or fail to initialize
                # User can decide whether to continue with remaining MCP servers
                logger.warning(
                    "Failed to initialize MCP server '%s' (URL: %s): %s. "
                    "Continuing without this server - tools will be unavailable.",
                    server_name,
                    server_url,
                    e,
                )

        return tools

    async def _json_tools(self) -> str:
        """
        Return tool information in OpenAI format with server configuration metadata.
        Each tool includes its server URL in the description.
        Returns JSON array of tools in OpenAI function-calling format.
        Parallelizes tool listing across all servers to reduce latency.
        """
        # Create tasks to list tools from all servers in parallel
        server_items = list(self._servers.items())  # pylint: disable=no-member
        tasks = [
            self._list_tools_from_server(server_name, server_config)
            for server_name, server_config in server_items
        ]

        # Gather results from all servers concurrently
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Merge tools, keeping first occurrence of duplicates
        all_tools: List[Dict[str, Any]] = []
        seen_tool_names: set = set()

        # Build URL to server name mapping for logging
        url_to_name = {
            config.url: name
            for name, config in self._servers.items()  # pylint: disable=no-member
        }

        for tools_from_server in results:
            for tool in tools_from_server:
                tool_name = tool["function"]["name"]
                server_url = tool.get("server_url", "")
                server_name = url_to_name.get(server_url, "unknown")

                # Skip duplicate tool names - keep first occurrence
                if tool_name in seen_tool_names:
                    logger.warning(
                        "Duplicate tool name '%s' from server '%s' (URL: %s). "
                        "Skipping this tool to avoid conflicts.",
                        tool_name,
                        server_name,
                        server_url,
                    )
                    continue

                seen_tool_names.add(tool_name)
                # Track which server provides this tool
                self._tool_to_server[tool_name] = server_name
                all_tools.append(tool)

        # Log summary of loaded tools
        total_tools = len(all_tools)
        total_servers = len(self._servers)

        # Determine which servers succeeded and which failed
        # server_url is at the top level of the tool dict
        successful_server_urls = set(
            tool.get("server_url") for tool in all_tools if tool.get("server_url")
        )
        successful_server_names = [
            name
            for name, config in self._servers.items()  # pylint: disable=no-member
            if config.url in successful_server_urls
        ]
        failed_server_names = [
            name
            for name, config in self._servers.items()  # pylint: disable=no-member
            if config.url not in successful_server_urls
        ]

        logger.info(
            "MCP initialization complete: Loaded %d tools from %d/%d servers",
            total_tools,
            len(successful_server_names),
            total_servers,
        )

        if successful_server_names:
            logger.info(
                "✓ Available servers: %s",
                ", ".join(f"'{name}'" for name in successful_server_names),
            )

        if failed_server_names:
            logger.warning(
                "✗ Unavailable servers: %s",
                ", ".join(f"'{name}'" for name in failed_server_names),
            )

        # Return tools in OpenAI format
        return json.dumps(all_tools)

    async def _invoke_tool(
        self, tool_name: str, arguments: Dict[str, Any], server_url: str
    ) -> str:
        """
        Execute a single tool call on the specified server.

        Args:
            tool_name: The name of the tool to invoke
            arguments: The arguments to pass to the tool
            server_url: The URL of the server that provides this tool
        """
        # Find the server config by URL
        server_config = None
        server_name = None

        for name, config in self._servers.items():  # pylint: disable=no-member
            if config.url == server_url:
                server_config = config
                server_name = name
                break

        if not server_config or not server_name:
            error_msg = f"Server with URL '{server_url}' not found in configuration"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        async with self._server_locks[server_name]:
            try:
                async with _session_context(
                    server_config.url,
                    server_config.type,
                    f"calling tool '{tool_name}' on {server_name}",
                    headers=server_config.headers,
                ) as session:
                    logger.info(
                        "Calling tool '%s' on server '%s' (URL: %s)",
                        tool_name,
                        server_name,
                        server_config.url,
                    )
                    result = await session.call_tool(tool_name, arguments)

                    parts: List[str] = []
                    for part in result.content or []:  # type: ignore[attr-defined]
                        text_payload = getattr(part, "text", None)
                        if isinstance(text_payload, str) and text_payload:
                            parts.append(text_payload)
                            continue

                        json_payload = getattr(part, "json", None)
                        if json_payload is not None:
                            try:
                                parts.append(json.dumps(json_payload))
                            except TypeError:
                                parts.append(str(json_payload))

                    return "\n".join(parts) if parts else str(result)

            except Exception as e:
                error_msg = (
                    f"Failed to call tool '{tool_name}' on server '{server_name}' "
                    f"(URL: {server_config.url}): {e}"
                )
                logger.error(error_msg)
                return json.dumps({"error": str(e)})

    async def __call__(self, request_id: str, **kwargs):
        action = kwargs.pop("action", None)
        if action not in {"list_tools", "call_tool"}:
            payload = json.dumps({"error": f"Unknown action '{action}'"})
        else:
            try:
                if action == "list_tools":
                    payload = await self._json_tools()
                else:  # call_tool
                    tool_name: Optional[str] = kwargs.get("tool_name")
                    if not tool_name:
                        raise ValueError("'tool_name' missing for call_tool action")
                    server_url: Optional[str] = kwargs.get("server_url")
                    if not server_url:
                        raise ValueError("'server_url' missing for call_tool action")
                    arguments: Dict[str, Any] = kwargs.get("arguments", {})
                    payload = await self._invoke_tool(tool_name, arguments, server_url)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("MCPExecutor error during '%s'", action)
                payload = json.dumps({"error": str(exc)})
        response_request_args = DistillerMessageRequestArgs(content=payload)
        response_payload = DistillerOutgoingMessage(
            account=self.account,
            project=self.project,
            uuid=self.uuid,
            role=self.role,
            request_id=request_id,
            request_type=DistillerMessageRequestType.EXECUTOR,
            request_args=response_request_args,
        )

        await self.send_queue.put(response_payload)
        return payload
