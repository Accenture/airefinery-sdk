"""Tool Use Agent Configuration Schema"""

from typing import List

from pydantic import BaseModel, Field


class ToolUseConfig(BaseModel):
    """
    Configuration for a Tool Use Agent
    """

    builtin_tools: List[str] = Field(
        default_factory=list,
        description="List of built-in tool names available to the agent.",
    )
    custom_tools: List[str] = Field(
        default_factory=list,
        description="List of JSON-encoded custom tool definitions.",
    )
    wait_time: int = Field(
        default=600,
        description="Maximum wait time (in seconds) for tool execution before timing out.",
    )
    enable_interpreter: bool = Field(
        default=False,
        description="Whether to post-process tool outputs with the interpreter agent.",
    )
