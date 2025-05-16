from typing import List, Tuple
from abc import ABCMeta, abstractmethod

from pydantic import BaseModel, Field

from air.knowledge.schema import Document
from air.knowledge.pipeline.chunking.chunking_registry import ChunkingRegistry


class ChunkingConfig(BaseModel):
    """
    Chunking configuration class
    """

    algorithm: str = Field(
        default="BruteForceChunking", description="Type of Chunking Algorithm"
    )
    chunk_size: int = Field(..., description="Max length per chunk")
    overlap_size: int = Field(
        default=0, description="Overlap between two neighboring chunks"
    )


class ChunkingMeta(ABCMeta):
    """
    A metaclass that registers any concrete subclass of BaseChunking
    in ChunkingRegistry at creation time.

    Because BaseChunking already depends on ABC (which uses ABCMeta),
    we must inherit from ABCMeta here to avoid a metaclass conflict.
    """

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)

        # Avoid registering the abstract base itself or any other classes
        # that are still abstract (i.e., if they haven't implemented all
        # abstract methods).
        if cls.__name__ != "BaseChunking" and not getattr(
            cls, "__abstractmethods__", False
        ):
            ChunkingRegistry.register(cls)


class BaseChunking(metaclass=ChunkingMeta):
    """
    Base class for chunking strategies.
    """

    def __init__(self, chunking_config: ChunkingConfig):
        self.chunk_size = chunking_config.chunk_size
        self.overlap_size = chunking_config.overlap_size

    @abstractmethod
    def run(self, documents: List[Document]) -> Tuple[List[Document], bool]:
        """
        Chunk a list of documents and return a list of chunked documents
        """
