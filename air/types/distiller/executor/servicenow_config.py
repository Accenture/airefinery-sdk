"""ServiceNow Agent Configuration Schema"""

from typing import Dict, List, Optional

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


class Provider(BaseModel):
    organization: str = Field(
        default="",
        description="Name of the organization providing the agent.",
    )
    url: str = Field(
        default="",
        description="URL of the organization.",
    )


class AuthorizationCodeFlow(BaseModel):
    authorizationUrl: str = Field(
        default="",
        description="URL for authorization.",
    )
    refreshUrl: str = Field(
        default="",
        description="URL for token refresh.",
    )
    scopes: Dict[str, str] = Field(
        default_factory=dict,
        description="Scopes for the authorization flow.",
    )
    tokenUrl: str = Field(
        default="",
        description="URL for token retrieval.",
    )


class ClientCredentialsFlow(BaseModel):
    refreshUrl: str = Field(
        default="",
        description="URL for token refresh.",
    )
    scopes: Dict[str, str] = Field(
        default_factory=dict,
        description="Scopes for the client credentials flow.",
    )
    tokenUrl: str = Field(
        default="",
        description="URL for token retrieval.",
    )


class OAuthFlows(BaseModel):
    authorizationCode: AuthorizationCodeFlow = Field(
        default_factory=AuthorizationCodeFlow,
        description="Authorization code flow details.",
    )
    clientCredentials: ClientCredentialsFlow = Field(
        default_factory=ClientCredentialsFlow,
        description="Client credentials flow details.",
    )


class OAuthScheme(BaseModel):
    flows: OAuthFlows = Field(
        default_factory=OAuthFlows,
        description="OAuth flows supported by the agent.",
    )
    type: str = Field(
        default="",
        description="Type of OAuth scheme.",
    )
    oauth2MetadataUrl: str = Field(
        default="",
        description="Metadata URL for OAuth2.",
    )


class SecuritySchemes(BaseModel):
    oauth: OAuthScheme = Field(
        default_factory=OAuthScheme,
        description="OAuth security scheme details.",
    )


class Capabilities(BaseModel):
    streaming: bool = Field(
        default_factory=bool,
        description="Indicates if streaming is supported.",
    )
    pushNotifications: bool = Field(
        default_factory=bool,
        description="Indicates if push notifications are supported.",
    )
    stateTransitionHistory: bool = Field(
        default_factory=bool,
        description="Indicates if state transition history is supported.",
    )


class Skill(BaseModel):
    id: str = Field(
        default="",
        description="Unique identifier for the skill.",
    )
    name: str = Field(
        default="",
        description="Name of the skill.",
    )
    description: str = Field(
        default="",
        description="Description of the skill.",
    )
    inputModes: List[str] = Field(
        default_factory=list,
        description="Supported input modes for the skill.",
    )
    outputModes: List[str] = Field(
        default_factory=list,
        description="Supported output modes for the skill.",
    )
    security: List[Dict] = Field(
        default_factory=list,
        description="Security details for the skill.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags associated with the skill.",
    )


class ServiceNowAgentCard(BaseModel):
    protocolVersion: str = Field(
        default="",
        description="Protocol version used by the agent.",
    )
    name: str = Field(
        default="",
        description="Name of the agent.",
    )
    description: str = Field(
        default="",
        description="Description of the agent.",
    )
    url: str = Field(
        default="",
        description="URL of the agent.",
    )
    preferredTransport: str = Field(
        default="",
        description="Preferred transport method for the agent.",
    )
    additionalInterfaces: List[str] = Field(
        default_factory=list,
        description="Additional interfaces supported by the agent.",
    )
    provider: Provider = Field(
        default_factory=Provider,
        description="Provider details for the agent.",
    )
    iconUrl: str = Field(
        default="",
        description="URL for the agent's icon.",
    )
    version: str = Field(
        default="",
        description="Version of the agent.",
    )
    documentationUrl: str = Field(
        default="",
        description="URL for the agent's documentation.",
    )
    capabilities: Capabilities = Field(
        default_factory=Capabilities,
        description="Capabilities of the agent.",
    )
    securitySchemes: SecuritySchemes = Field(
        default_factory=SecuritySchemes,
        description="Security schemes supported by the agent.",
    )
    security: List[Dict[str, List[str]]] = Field(
        default_factory=list,
        description="Security details for the agent.",
    )
    defaultInputModes: List[str] = Field(
        default_factory=list,
        description="Default input modes supported by the agent.",
    )
    defaultOutputModes: List[str] = Field(
        default_factory=list,
        description="Default output modes supported by the agent.",
    )
    skills: List[Skill] = Field(
        default_factory=list,
        description="Skills supported by the agent.",
    )
    supportsAuthenticatedExtendedCard: bool = Field(
        default_factory=bool,
        description="Indicates if authenticated extended cards are supported.",
    )
    signatures: List[Dict] = Field(
        default_factory=list,
        description="Signatures associated with the agent.",
    )
