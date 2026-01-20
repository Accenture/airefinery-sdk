from __future__ import annotations

from http import HTTPStatus
from typing import Any

__all__ = [
    "DistillerClientError",
    "AuthenticationError",
    "ProjectCreationError",
    "ProjectDownloadError",
    "ConnectionError",
    "UserAlreadyConnectedError",
    "WebSocketError",
    "WebSocketSendError",
    "WebSocketReceiveError",
    "ConnectionTimeoutError",
    "ConnectionClosedError",
    "DatabaseError",
    "HistoryRetrievalError",
    "ChatLoggingError",
]


class DistillerClientError(Exception):
    """Base exception for all distiller client errors."""

    DEFAULT_CODE = "distiller.client.error"
    DEFAULT_STATUS = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: str | None = None,
        status: HTTPStatus | None = None,
        extra: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message or "An error occurred in the distiller client"
        self.error_code = error_code or self.DEFAULT_CODE
        self.status = status or self.DEFAULT_STATUS
        self.extra = extra or {}
        if cause:
            self.__cause__ = cause


class AuthenticationError(DistillerClientError):
    """Raised when API key validation fails."""

    DEFAULT_CODE = "distiller.client.authentication_error"
    DEFAULT_STATUS = HTTPStatus.UNAUTHORIZED


class ProjectCreationError(DistillerClientError):
    """Raised when project creation fails."""

    DEFAULT_CODE = "distiller.client.project_creation_error"
    DEFAULT_STATUS = HTTPStatus.BAD_REQUEST


class ProjectDownloadError(DistillerClientError):
    """Raised when project download fails."""

    DEFAULT_CODE = "distiller.client.project_download_error"
    DEFAULT_STATUS = HTTPStatus.NOT_FOUND


class ConnectionError(DistillerClientError):
    """Base class for connection-related errors."""

    DEFAULT_CODE = "distiller.client.connection_error"
    DEFAULT_STATUS = HTTPStatus.SERVICE_UNAVAILABLE


class WebSocketError(ConnectionError):
    """Base class for WebSocket-related errors."""

    DEFAULT_CODE = "distiller.client.websocket_error"


class WebSocketSendError(WebSocketError):
    """Raised when sending a message via WebSocket fails."""

    DEFAULT_CODE = "distiller.client.websocket_send_error"


class WebSocketReceiveError(WebSocketError):
    """Raised when receiving a message via WebSocket fails."""

    DEFAULT_CODE = "distiller.client.websocket_receive_error"


class ConnectionTimeoutError(ConnectionError):
    """Raised when connection times out (e.g., ping monitor)."""

    DEFAULT_CODE = "distiller.client.connection_timeout"
    DEFAULT_STATUS = HTTPStatus.GATEWAY_TIMEOUT


class ConnectionClosedError(ConnectionError):
    """Raised when WebSocket connection is closed unexpectedly."""

    DEFAULT_CODE = "distiller.client.connection_closed"


class UserAlreadyConnectedError(ConnectionError):
    """Raised when the user is already connected with the same credentials."""

    DEFAULT_CODE = "distiller.client.user_already_connected"
    DEFAULT_STATUS = HTTPStatus.CONFLICT


class DatabaseError(DistillerClientError):
    """Base class for database-related errors."""

    DEFAULT_CODE = "distiller.client.database_error"
    DEFAULT_STATUS = HTTPStatus.INTERNAL_SERVER_ERROR


class HistoryRetrievalError(DatabaseError):
    """Raised when retrieving chat history fails."""

    DEFAULT_CODE = "distiller.client.history_retrieval_error"


class ChatLoggingError(DatabaseError):
    """Raised when logging chat to database fails."""

    DEFAULT_CODE = "distiller.client.chat_logging_error"
