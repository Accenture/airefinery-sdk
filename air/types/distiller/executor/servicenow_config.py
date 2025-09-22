"""ServiceNow Agent Configuration Schema"""

from pydantic import BaseModel, Field, model_validator


class PublicAgentCardConfig(BaseModel):
    """
    Public agent card details.
    Two fields, both required.
    """

    public_agent_card_path: str = Field(
        default="", description="Path to the public agent card."
    )
    rpc_url: str = Field(default="", description="RPC URL for the agent.")

    @model_validator(mode="after")
    def check_public_agent_card_fields(self):
        public_path = self.public_agent_card_path
        rpc = self.rpc_url

        if public_path == "" or rpc == "":
            raise ValueError(
                "Both 'public_agent_card_path' and 'rpc_url' must be provided."
            )
        return self


class AgentCardConfig(BaseModel):
    """
    Dictionary containing the public agent card information.
    """

    public: PublicAgentCardConfig = Field(
        default=PublicAgentCardConfig(
            public_agent_card_path="dummy_path", rpc_url="dummy_url"
        ),
        description="Public agent card details.",
    )

    @model_validator(mode="after")
    def check_public_agent_card_non_empty(self):
        public = self.public

        if not public:
            raise ValueError("Public agent card details must be provided.")
        return self


class ServiceNowAgentConfig(BaseModel):
    """
    ServiceNow Agent Config
    """

    servicenow_token: str = Field(
        default="",
        description="Name of the environment variable containing the ServiceNow API token.",
    )
    agent_card: AgentCardConfig = Field(
        default_factory=AgentCardConfig,
        description="Dictionary containing the public agent card information "
        "(agent card path and RPC URL).",
    )
    wait_time: int = Field(
        default=300,
        description="Time in seconds to wait for a response from the "
        "ServiceNow agent before timing out.",
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
        servicenow_token = self.servicenow_token

        if servicenow_token == "":
            raise ValueError(
                "Missing 'servicenow_token' in utility_config for ServiceNowExecutor."
            )

        return self
