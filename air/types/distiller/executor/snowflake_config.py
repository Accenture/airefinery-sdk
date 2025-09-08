"""Snowflake Agent Configuration Schema"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field, model_validator


class SearchServiceConfig(BaseModel):
    name: str = Field(
        default="",
        description="Arbitrary name for the Cortex Search service."
        "This name is used for referring to this service in your developments.",
    )
    database: str = Field(
        default="",
        description="Name of the database that hosts the Cortex Search service.",
    )
    db_schema: str = Field(
        default="",
        description="Name of the schema that hosts the Cortex Search service.",
    )
    service_name: str = Field(
        default="",
        description="Name of the Cortex Search service as recorded on the Snowflake platform.",
    )


class AnalystServiceConfig(BaseModel):
    name: str = Field(
        default="",
        description="Arbitrary name for the Cortex Analyst service. "
        "This name is used for referring to this service in your developments.",
    )
    database: str = Field(
        default="",
        description="Name of the database that hosts the Cortex Analyst service.",
    )
    db_schema: str = Field(
        default="",
        description="Name of the schema that hosts the Cortex Analyst service.",
    )
    stage: str = Field(
        default="",
        description="Name of the stage that hosts the Cortex Analyst service.",
    )
    file_name: str = Field(
        default="",
        description="Name of the semantic file that corresponds to the Cortex Analyst service.",
    )
    warehouse: str = Field(
        default="",
        description="Name of the warehouse that hosts the Cortex Analyst service.",
    )
    user_role: str = Field(
        default="",
        description="User role that has access to the Cortex Analyst service, "
        "and correspondingly to the Cortex agent.",
    )


class SnowflakeServicesConfig(BaseModel):
    search: List[SearchServiceConfig] = Field(
        default_factory=list,
        description="List of the required Cortex Search services.",
    )
    analyst: List[AnalystServiceConfig] = Field(
        default_factory=list,
        description="List of the required Cortex Analyst services.",
    )


class SnowflakeAgentConfig(BaseModel):
    """
    Snowflake Agent Config
    """

    snowflake_password: str = Field(
        default="",
        description="The environment variable containing the Snowflake "
        "Programmatic Access Token (PAT) for ADMIN role.",
    )
    snowflake_services: SnowflakeServicesConfig = Field(
        default_factory=SnowflakeServicesConfig,
        description="Search and Analyst services definitions.",
    )
    snowflake_model: str = Field(
        default="",
        description="Underlying LLM model of the Cortex Agent.",
    )
    snowflake_base_url: str = Field(
        default="",
        description="The base URL of where the Cortex Agent is located.",
    )
    sql_timeout: int = Field(
        default=10,
        description="Timeout in seconds for SQL statement execution",
    )
    system_prompt: str = Field(
        default="",
        description="The system instruction for Cortex Agent.",
    )
    snowflake_experimental: Dict[str, Any] = Field(
        default_factory=dict,
        description="Experimental flags passed to the Cortex Agent.",
    )
    snowflake_tool_choice: str = Field(
        default="auto",
        description="The configuration type used to select the Snowflake tools.",
    )
    thought_process_tracing: bool = Field(
        default=False,
        description="Configure tracing of thought process of the agent.",
    )
    contexts: list = Field(
        default_factory=list,
        description="List of additional contexts to be passed to the agent.",
    )

    @model_validator(mode="after")
    def check_connection_params_non_empty(self):
        """
        Checking if required connection parameters are populated in the config
        """
        snowflake_password = self.snowflake_password
        snowflake_model = self.snowflake_model
        snowflake_base_url = self.snowflake_base_url

        if snowflake_password == "":
            raise ValueError(
                "Missing 'snowflake_password' in utility_config for SnowflakeExecutor."
            )

        if snowflake_model == "":
            raise ValueError(
                "Missing 'snowflake_model' in utility_config for SnowflakeExecutor."
            )

        if snowflake_base_url == "":
            raise ValueError(
                "Missing 'snowflake_base_url' in utility_config for SnowflakeExecutor."
            )

        return self

    @model_validator(mode="after")
    def check_connection_params_valid_values(self):
        """
        Checking if some of the connection parameters are populated from valid values
        """
        snowflake_model = self.snowflake_model
        snowflake_tool_choice = self.snowflake_tool_choice
        valid_models = [
            "llama3.1-70b",
            "llama3.3-70b",
            "mistral-large2",
            "claude-3-5-sonnet",
        ]
        valid_tool_choices = ["auto", "required", "tool"]

        if snowflake_model not in valid_models:
            raise ValueError("Input value for 'snowflake_model' is not valid.")

        if snowflake_tool_choice not in valid_tool_choices:
            raise ValueError("Input value for 'snowflake_tool_choice' is not valid.")

        return self
