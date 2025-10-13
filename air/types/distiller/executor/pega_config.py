"""Pega Agent Configuration Schema"""

from pydantic import Field, BaseModel, model_validator


class PegaAgentConfig(BaseModel):
    """
    Pega Agent Config
    - client_id: Required, must be non-empty
    - client_secret: Required, must be non-empty
    - token_url: Required, must be non-empty
    - base_url: Required, must be non-empty
    - wait_time: Optional, integer.
    - contexts: Optional, list
    """

    client_id: str = Field(
        default="",
        description="Name of the environment variable containing the Pega Client ID.",
    )
    client_secret: str = Field(
        default="",
        description="Name of the environment variable containing the Pega Client Secret.",
    )
    token_url: str = Field(
        default="", description="The token URL where the agents use to authenticate."
    )
    base_url: str = Field(
        default="", description="The base URL where the agents are located."
    )
    wait_time: int = Field(
        default=300,
        description="Time in seconds to wait for a response from the "
        "Pega agent before timing out.",
    )
    contexts: list = Field(
        default_factory=list,
        description="List of additional contexts to be passed to the agent.",
    )

    @model_validator(mode="after")
    def check_empty_connection_params(self):
        client_id = self.client_id
        client_secret = self.client_secret
        token_url = self.token_url
        base_url = self.base_url

        if client_id == "":
            raise ValueError("Missing 'client_id' in utility_config for PegaExecutor.")

        if client_secret == "":
            raise ValueError(
                "Missing 'client_secret' in utility_config for PegaExecutor."
            )

        if token_url == "":
            raise ValueError("Missing 'token_url' in utility_config for PegaExecutor.")

        if base_url == "":
            raise ValueError("Missing 'base_url' in utility_config for PegaExecutor.")
        return self
