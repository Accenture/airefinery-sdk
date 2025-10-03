"""Wolfram Agent Configuration Schema"""

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class WolframAgentConfig(BaseModel):
    """
    Wolfram Agent Config
    """

    app_id: str = Field(
        default="",
        description="Name of the environment variable containing the Wolfram App ID.",
    )
    base_url: str = Field(
        default="https://www.wolframalpha.com/api/v1/llm-api",
        description="The base URL for the Wolfram API (LLM-optimized endpoint).",
    )
    timeout: int = Field(
        default=60,
        description="Timeout in seconds for API requests",
    )
    enable_interpreter: bool = Field(
        default=False,
        description="Enable interpretation layer to format responses in a more user-friendly way.",
    )
    output_format: List[
        Literal["raw", "text", "images", "wolfram_code", "website_link"]
    ] = Field(
        default=["text"],
        description="List of output components to include in response",
    )

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: List[str]) -> List[str]:
        """Ensure that output_format is a non-empty list with only valid options."""
        valid_options = ["text", "images", "website_link"]
        if not v:
            raise ValueError("output_format list cannot be empty")
        if not all(item in valid_options for item in v):
            raise ValueError(f"Invalid output format. Must be one of: {valid_options}")
        return v

    @model_validator(mode="before")
    @classmethod
    def reject_contexts_field(cls, data: dict) -> dict:
        """
        Ensure the config does not contain a 'contexts' field
        """
        if isinstance(data, dict) and "contexts" in data:
            raise ValueError("Field 'contexts' is not allowed in WolframAgentConfig")
        return data

    @model_validator(mode="after")
    def check_connection_params_non_empty(self):
        """
        Checking if required connection parameters are populated in the config
        """
        app_id = self.app_id
        base_url = self.base_url

        if app_id == "":
            raise ValueError("Missing 'app_id' in utility_config for WolframExecutor.")

        if base_url == "":
            raise ValueError(
                "Missing 'base_url' in utility_config for WolframExecutor."
            )

        return self
