# pylint: disable=too-few-public-methods
"""
Provides synchronous and asynchronous clients for knowledge-related APIs.

Includes:
- AsyncKnowledgeClient for accessing the knowledge graph.
- KnowledgeClient for document processing tasks.
"""
import asyncio
from functools import cached_property

from air.knowledge.document_processing_client import DocumentProcessingClient


class AsyncKnowledgeClient:
    """
    Asynchronous client for knowledge services, including Graph API.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_headers: dict[str, str] | None = None,
    ):
        """
        Initialize the async knowledge client.

        Args:
            base_url (str): API base URL.
            api_key (str): API key for authentication.
            default_headers (dict, optional): Additional headers.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers

    _lock = asyncio.Lock()

    @property
    def document_processing(self):
        """
        Document processing is not supported in async mode.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "Document processing is only available in synchronous mode. Use KnowledgeClient instead."
        )

    @cached_property
    def _graph(self):
        """
        Cached property `graph`
        """
        from air.knowledge.knowledge_graph_client import (  # pylint: disable=import-outside-toplevel
            KnowledgeGraphClient,
        )

        return KnowledgeGraphClient()

    async def get_graph(self):
        """
        Async safe method to get KnowledgeGraphClient object
        """
        async with self._lock:
            return self._graph


class KnowledgeClient:
    """
    Synchronous client for knowledge services, including document processing.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_headers: dict[str, str] | None = None,
    ):
        """
        Initialize the sync knowledge client.

        Args:
            base_url (str): API base URL.
            api_key (str): API key for authentication.
            default_headers (dict, optional): Additional headers.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers
        self.document_processing = DocumentProcessingClient(base_url=self.base_url)

    def get_graph(self):
        """
        Knowledge Graph is not supported in sync mode.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "Knowledge Graph is only available in asynchronous mode. Use AsyncKnowledgeClient instead."
        )
