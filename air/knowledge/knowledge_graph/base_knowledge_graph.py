"""
Base module for Knowledge Graph, supports building, updating, and querying
a knowledge graph.
"""

import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Union

from air.knowledge.knowledge_graph.knowledge_graph_registry import (
    KnowledgeGraphRegistry,
)
from air.types import KnowledgeGraphConfig

logger = logging.getLogger(__name__)


class KnowledgeGraphMeta(ABCMeta):
    """
    A metaclass that registers any concrete subclass of BaseKnowledgeGraph
    in KnowledgeGraphRegistry at creation time.

    Because BaseKnowledgeGraph already depends on ABC (which uses ABCMeta),
    we must inherit from ABCMeta here to avoid a metaclass conflict.
    """

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)

        # Avoid registering the abstract base itself or any other classes
        # that are still abstract (i.e., if they haven't implemented all
        # abstract methods).
        if cls.__name__ != "BaseKnowledgeGraph" and not getattr(
            cls, "__abstractmethods__", False
        ):
            KnowledgeGraphRegistry.register(cls)


class BaseKnowledgeGraph(metaclass=KnowledgeGraphMeta):
    """
    Base class for a knowledge graph.
    """

    def __init__(self, graph_config: KnowledgeGraphConfig):
        self.base_url = os.environ.get("KNOWLEDGE_GRAPH_API_BASE_URL")
        if not self.base_url:
            raise ValueError("KNOWLEDGE_GRAPH_API_BASE_URL env variable not set!")
        self.work_dir = graph_config.work_dir
        self.api_key = os.environ.get("KNOWLEDGE_GRAPH_API_KEY")
        if not self.api_key:
            raise ValueError("KNOWLEDGE_GRAPH_API_KEY env variable not set!")
        self.api_type = graph_config.api_type
        self.llm_model = graph_config.llm_model
        self.embedding_model = graph_config.embedding_model
        self.chunk_size = graph_config.chunk_size
        self.chunk_overlap = graph_config.chunk_overlap
        logger.info("Creating directory at: %s", self.work_dir)
        os.makedirs(self.work_dir, exist_ok=True)

    @abstractmethod
    async def build(self) -> bool:
        """
        Add an entity to the knowledge graph.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def update(self) -> bool:
        """
        Add a relationship to the knowledge graph.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def query(self, query, method: str) -> Union[str, None]:
        """
        Query the knowledge graph.
        """
        raise NotImplementedError("Subclasses should implement this method.")
