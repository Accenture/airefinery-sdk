"""CB Insights Agent Configuration Schema"""

from pydantic import BaseModel, Field, model_validator


class CBInsightsAgentConfig(BaseModel):
    """
    CB Insights Agent Config
    - client_id: Required, must be non-empty
    - client_secret: Required, must be non-empty
    - api_base_url: Required, must be non-empty
    - wait_time: Optional, integer
    - contexts: Optional, list
    """

    client_id: str = Field(
        default="",
        description="Name of the environment variable containing the CB Insights Client ID.",
    )
    client_secret: str = Field(
        default="",
        description=(
            "Name of the environment variable containing the CB Insights Client Secret."
        ),
    )
    api_base_url: str = Field(
        default="",
        description="The base URL for the CB Insights API.",
    )
    wait_time: int = Field(
        default=300,
        description=(
            "Time in seconds to wait for a response from the "
            "CB Insights API before timing out."
        ),
    )
    contexts: list = Field(
        default_factory=list,
        description="List of additional contexts to be passed to the agent.",
    )

    @model_validator(mode="after")
    def check_empty_connection_params(self):
        """
        Validate that required connection parameters are not empty.
        """
        client_id = self.client_id
        client_secret = self.client_secret
        api_base_url = self.api_base_url

        if client_id == "":
            raise ValueError(
                "Missing 'client_id' in utility_config for CBInsightsExecutor."
            )

        if client_secret == "":
            raise ValueError(
                "Missing 'client_secret' in utility_config for CBInsightsExecutor."
            )

        if api_base_url == "":
            raise ValueError(
                "Missing 'api_base_url' in utility_config for CBInsightsExecutor."
            )
        return self
