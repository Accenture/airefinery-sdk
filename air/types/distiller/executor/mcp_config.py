"""MCP Client Agent Configuration Schema"""

import logging
import os
import re
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """
    Configuration for a single MCP server
    """

    url: str = Field(
        description=(
            "The URL for the MCP server endpoint "
            "(e.g., http://localhost:4003/mcp or http://localhost:4003/sse)"
        )
    )
    type: Literal["http-stream", "sse"] = Field(
        default="http-stream",
        description=(
            "The type of MCP server: 'http-stream' for streamable HTTP (recommended) "
            "or 'sse' for Server-Sent Events"
        ),
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Optional HTTP headers to send with requests (e.g., for authentication). "
            "Supports environment variable substitution with ENV:VAR_NAME syntax"
        ),
    )

    @field_validator("headers")
    @classmethod
    def substitute_env_variables(
        cls, v: Optional[Dict[str, str]]
    ) -> Optional[Dict[str, str]]:
        """
        Substitute environment variables in header values.
        Supports ENV:VAR_NAME syntax for environment variable substitution.
        Example: ENV:NPS_API_KEY will be replaced with the value of NPS_API_KEY from os.environ
        Headers can be None or empty dict if not needed.
        """
        # Allow headers to be None or empty
        if v is None or not v:
            return v

        processed_headers = {}
        env_prefix_pattern = re.compile(r"^ENV:(.+)$")

        for key, value in v.items():
            if isinstance(value, str):
                # Check for ENV: prefix pattern
                env_match = env_prefix_pattern.match(value)
                if env_match:
                    var_name = env_match.group(1)
                    env_value = os.environ.get(var_name)
                    if not env_value:
                        # Raise error since headers are for authentication/necessary values
                        raise ValueError(
                            f"Environment variable '{var_name}' not found or empty for header '{key}'. "
                            f"Please set the environment variable or remove the header. "
                            f"Use ENV:VAR_NAME syntax (e.g., ENV:{var_name})"
                        )
                    processed_headers[key] = env_value
                else:
                    # No substitution needed, use value as-is
                    processed_headers[key] = value
            else:
                processed_headers[key] = value

        return processed_headers

    model_config = {"extra": "forbid"}


class MCPClientAgentConfig(BaseModel):
    """
    MCPClient Agent Config
    """

    # New multi-server configuration
    mcpServers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="Dictionary of MCP server configurations, keyed by server name",
    )

    # Legacy single-server configuration (backward compatibility)
    mcp_sse_url: str = Field(
        default="",
        description=(
            "The URL for MCP SSE endpoint (e.g., http://localhost:4003/sse). "
            "Deprecated: Use mcpServers for new implementations. "
            "This field is kept only for backward compatibility with existing configurations."
        ),
    )
    enable_interpreter: bool = Field(
        default=False,
        description=(
            "The setting controlling the use of an interpreter that makes the response "
            "of the MCP server more user-friendly."
        ),
    )
    tool_call_interval: int = Field(
        default=1,
        description="The time in seconds between consecutive tool calls.",
    )
    max_tool_calls: int = Field(
        default=5,
        description="The maximum number of tool calls.",
    )
    show_tool_progress: bool = Field(
        default=False,
        description="The setting controlling whether tool call progress will be shown.",
    )
    wait_time: int = Field(
        default=300,
        description="The time in seconds to wait for the MCP server's response.",
    )

    @model_validator(mode="before")
    @classmethod
    def check_connection_params_non_empty(cls, values: dict):
        """
        Validate connection configuration before model initialization.

        - Supports both multi-server (mcpServers) and single-server (mcp_sse_url) configs
        - Ensures at least one MCP server is configured
        - Converts single-server config to multi-server format internally
        - mcp_sse_url automatically uses 'sse' type for backward compatibility
        """
        # Check for new multi-server configuration
        mcp_servers = values.get("mcpServers") or {}

        # Check for previous legacy single-server configuration
        mcp_sse_url = (values.get("mcp_sse_url") or "").strip()

        # --- 1. Handle case where both configurations are provided ---
        if mcp_servers and mcp_sse_url:
            logger.warning(
                "Both 'mcpServers' and legacy 'mcp_sse_url' are configured. "
                "Prioritizing 'mcpServers' and ignoring 'mcp_sse_url'."
            )
            # Continue with mcpServers, ignore mcp_sse_url

        # --- 2. Handle multi-server configuration ---
        if mcp_servers:
            if not isinstance(mcp_servers, dict) or not mcp_servers:
                raise ValueError("'mcpServers' must be a non-empty dictionary")

            # Check for duplicate server names (case-insensitive)
            server_names_lower = {}
            for server_name in mcp_servers.keys():
                name_lower = server_name.lower()
                if name_lower in server_names_lower:
                    raise ValueError(
                        f"Duplicate server name detected: '{server_name}' conflicts with "
                        f"'{server_names_lower[name_lower]}'. Server names must be unique."
                    )
                server_names_lower[name_lower] = server_name

            # Validate and process each server configuration
            # MCPServerConfig will handle header processing automatically
            processed_servers = {}
            for server_name, server_config in mcp_servers.items():
                if not isinstance(server_config, dict):
                    raise ValueError(
                        f"Server '{server_name}' config must be a dictionary"
                    )
                # Instantiate MCPServerConfig to validate and process headers
                processed_servers[server_name] = MCPServerConfig(**server_config)

            values["mcpServers"] = processed_servers
            return values

        # --- 3. Handle previous legacy single-server configuration ---
        if not mcp_sse_url:
            raise ValueError(
                "Missing MCP server configuration. Provide either 'mcpServers' (recommended) "
                "or legacy single-server config 'mcp_sse_url'."
            )

        # Convert single-server to multi-server format internally
        logger.warning(
            "'mcp_sse_url' is still supported for backward compatibility. "
            "Please migrate to 'mcpServers' format for better multi-server support. "
            "Example: mcpServers: {'service_name': {'url': '%s', 'type': 'sse'}}",
            mcp_sse_url,
        )

        # Create MCPServerConfig instance for legacy SSE configuration
        server_config = MCPServerConfig(
            url=mcp_sse_url,
            type="sse",  # mcp_sse_url always uses SSE type
        )

        # Create mcpServers with a default server
        values["mcpServers"] = {"unknown_legacy_sse": server_config}

        return values
