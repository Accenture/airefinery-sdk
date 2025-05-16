import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.sse import sse_client

from air.distiller.executor.executor import Executor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _session_context(sse_url: str):
    """
    Create a short‑lived MCP ClientSession bound to a single coroutine.
    """
    async with sse_client(sse_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


class MCPExecutor(Executor):
    """
    Executor for MCPClientAgent.
    """

    agent_class: str = "MCPClientAgent"

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
    ) -> None:
        self._sse_url: str = utility_config.get("mcp_sse_url", "")
        if not self._sse_url:
            raise ValueError("'mcp_sse_url' missing from utility_config")

        super().__init__(
            func={},
            send_queue=send_queue,
            account=account,
            project=project,
            uuid=uuid,
            role=role,
            return_string=return_string,
        )

        logger.info("MCPExecutor initialized for %s", self._sse_url)

    async def _json_tools(self) -> str:
        """
        Return the remote tool list in OpenAI function‑calling format.
        """
        async with _session_context(self._sse_url) as session:
            tools_response = await session.list_tools()
            formatted: List[Dict[str, Any]] = []

            for tool in tools_response.tools:
                params_raw = tool.inputSchema or {}
                if not isinstance(params_raw, dict):
                    logger.warning(
                        "Tool %s inputSchema isn't dict – coercing to object", tool.name
                    )
                    params_raw = {}

                # Normalise to JSON‑Schema object form
                if "type" not in params_raw:
                    params_raw = {"type": "object", "properties": params_raw}
                params_raw.setdefault("properties", {})
                params_raw.setdefault("required", list(params_raw["properties"].keys()))

                formatted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description
                            or f"Tool '{tool.name}' on {self._sse_url}",
                            "parameters": params_raw,
                        },
                    }
                )

            return json.dumps(formatted)

    async def _invoke_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a single tool call.
        """
        async with _session_context(self._sse_url) as session:
            logger.info("Calling remote MCP tool '%s'", name)
            result = await session.call_tool(name, arguments)

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

    async def __call__(self, request_id: str, *args, **kwargs):
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
                    arguments: Dict[str, Any] = kwargs.get("arguments", {})
                    payload = await self._invoke_tool(tool_name, arguments)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("MCPExecutor error during '%s'", action)
                payload = json.dumps({"error": str(exc)})

        await self.send_queue.put(
            {
                "account": self.account,
                "project": self.project,
                "uuid": self.uuid,
                "role": self.role,
                "request_id": request_id,
                "request_type": "executor",
                "request_args": {"content": payload},
            }
        )
        return payload
