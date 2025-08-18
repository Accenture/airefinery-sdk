"""SAP Agent Configuration Schema"""

from pydantic import BaseModel, Field, model_validator


class SAPAgentConfig(BaseModel):
    """
    SAP Agent Config
    - url: Required, must be non-empty
    - contexts: Optional, list
    """

    url: str = Field(default="", description="The URL where the agent is located.")
    contexts: list = Field(
        default_factory=list,
        description="List of additional contexts to be passed to the agent.",
    )

    @model_validator(mode="after")
    def check_empty_url(self):
        url = self.url

        if url == "":
            raise ValueError("Missing 'url' in utility_config for SAPExecutor.")
        return self
