from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field

from air.types.base import CustomBaseModel


class DistillerMessageRequestType(str, Enum):
    """
    Type of distiller message request, among 'query', 'executor'.
    """

    QUERY = "query"
    EXECUTOR = "executor"


class DistillerMessageRequestArgs(CustomBaseModel, extra="allow"):
    """
    Query request args.
    """

    query: str = Field(default="", description="The query string to be processed.")
    image: Optional[str] = Field(
        default=None, description="Optional image to be included in the query."
    )
    content: Optional[Union[str, List]] = Field(
        default=None, description="Optional content to be included in the query."
    )


class DistillerOutgoingMessage(CustomBaseModel):
    """
    Message sent from the client to the server.
    """

    account: str = Field(
        default="", description="The account associated with the message."
    )
    request_type: DistillerMessageRequestType = Field(
        description="Type of request, among 'query', 'executor'."
    )
    project: str = Field(
        default="", description="The project associated with the message."
    )
    uuid: str = Field(
        default="", description="The unique identifier for the user making the request."
    )
    request_args: DistillerMessageRequestArgs | None = Field(
        default=None, description="Arguments for the message request."
    )
    role: str = Field(
        default="user", description="The role of the user making the request."
    )
    request_id: str = Field(
        default="single_request", description="The unique identifier for the request."
    )

    def __getitem__(self, key: str):
        return getattr(self, key)

    @classmethod
    def allowed_request_types(cls) -> tuple[str, ...]:
        """
        Tuple of all valid literals for `request_type`.
        """
        return tuple(item.value for item in DistillerMessageRequestType)


class DistillerIncomingMessage(CustomBaseModel, extra="allow"):
    """
    Message sent from the server to the client in response to a query.
    """

    role: str = Field(
        default="assistant", description="The role of the message sender."
    )
    content: str = Field(default="", description="The content of the message.")
    status: str = Field(default="", description="The status of the message.")
    image: Optional[dict] = Field(
        default=None, description="Optional image associated with the message."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default={}, description="Metadata associated with the message."
    )
    video: Optional[str] = Field(
        default=None, description="Optional video associated with the message."
    )
    chart_response: Optional[List[Any]] = Field(
        default_factory=list,
        description="List of chart responses associated with the message.",
    )
    chart_type: Optional[List[Any]] = Field(
        default_factory=list,
        description="List of chart types associated with the message.",
    )
    table_data: Optional[str] = Field(
        default=None, description="Optional table data associated with the message."
    )
    table_columns: Optional[str] = Field(
        default=None, description="Optional table columns associated with the message."
    )
    title: Optional[str] = Field(
        default=None, description="Optional title associated with the message."
    )
    type: Optional[str] = Field(
        default=None, description="Optional type associated with the message."
    )
    timestamp: Optional[float] = Field(
        default=None, description="Optional timestamp associated with the message."
    )
    uuid_timestamp: Optional[str] = Field(
        default=None, description="Optional UUID timestamp associated with the message."
    )
    request_id: Optional[str] = Field(
        default="single_request", description="The unique identifier for the request."
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier associated with the message.",
    )
    kwargs: Dict[str, Any] = Field(
        default={},
        description="Additional keyword arguments associated with the message.",
    )

    def __getitem__(self, key: str):
        return getattr(self, key)


class DistillerPongMessage(CustomBaseModel):
    """
    Pong message sent in response to a ping.
    """

    type: Literal["PONG"] = Field(
        default="PONG", description="Type of the message, always 'PONG'."
    )


class DistillerMemoryRequestType(str, Enum):
    """
    Type of distiller memory request, among 'memory/add', 'memory/reset','memory/retrieve'.
    """

    ADD = "memory/add"
    RESET = "memory/reset"
    RETRIEVE = "memory/retrieve"


class DistillerMemoryRequestArgs(CustomBaseModel, extra="allow"):
    """
    Memory query request args.
    """

    source: str = Field(default="", description="The source of the memory query.")
    variables_dict: Dict[str, str] = Field(
        default_factory=dict, description="Variables to be used in the memory query."
    )
    n_rounds: int = Field(
        default=0, description="Number of rounds for the memory query."
    )
    max_context: int = Field(
        default=32000, description="Maximum context size for the memory query."
    )
    format: Literal["str", "json_string"] = Field(
        default="str", description="Format of the memory query response."
    )


class DistillerMemoryOutgoingMessage(CustomBaseModel):
    """
    Memory query request.
    """

    account: str = Field(
        default="", description="The account associated with the memory request."
    )
    request_type: DistillerMemoryRequestType = Field(
        description="Type of memory request, among 'memory/add', 'memory/reset','memory/retrieve'."
    )
    project: str = Field(
        default="", description="The project associated with the memory request."
    )
    uuid: str = Field(
        default="",
        description="The unique identifier for the user making the memory request.",
    )
    request_args: DistillerMemoryRequestArgs | None = Field(
        default=None, description="Arguments for the memory request."
    )
    role: str = Field(
        default="user", description="The role of the user making the memory request."
    )
    request_id: str = Field(
        default="single_request",
        description="Unique identifier for the memory request.",
    )

    def __getitem__(self, key: str):
        return getattr(self, key)

    @classmethod
    def allowed_request_types(cls) -> tuple[str, ...]:
        """
        Tuple of all valid literals for `request_type`.
        """
        return tuple(item.value for item in DistillerMemoryRequestType)
