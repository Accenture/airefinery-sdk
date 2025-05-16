"""
Base module for Vector DB, supports upload and search
"""

import logging
from abc import ABCMeta
from typing import List

from pydantic import BaseModel, Field

from air.api.vector_db.vectordb_registry import VectorDBRegistry

logger = logging.getLogger(__name__)


class VectorDBConfig(BaseModel):
    """
    VectorDB configuration class
    """

    type: str = Field(default="AzureAISearch", description="Type of the Vector DB")
    base_url: str = Field(..., description="Vector DB URL")
    api_key: str = Field(..., description="API key required to access the vector DB")
    api_version: str = Field(..., description="API Version")
    index: str = Field(..., description="Name of the vector db index")
    embedding_column: str = Field(
        default="text_vector",
        description="Name of the column in the vector db that stores embeddings for vector searches",
    )
    top_k: int = Field(
        default=1,
        description="Number of top results (k) to return from each vector search request",
    )
    content_column: List[str] = Field(
        default=[],
        description="List of columns from which content should be returned in search results",
    )
    timeout: int = Field(default=60, description="Vector DB POST request timeout")


class VectorDBMeta(ABCMeta):
    """
    A metaclass that registers any concrete subclass of BaseVectorDB
    in VectorDBRegistry at creation time.

    Because BaseVectorDB already depends on ABC (which uses ABCMeta),
    we must inherit from ABCMeta here to avoid a metaclass conflict.
    """

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)

        # Avoid registering the abstract base itself or any other classes
        # that are still abstract (i.e., if they haven't implemented all
        # abstract methods).
        if cls.__name__ != "BaseVectorDB" and not getattr(
            cls, "__abstractmethods__", False
        ):
            VectorDBRegistry.register(cls)


class BaseVectorDB(metaclass=VectorDBMeta):
    """
    Base configuration class for any vector DB
    """

    def __init__(self, vectordb_config: VectorDBConfig):
        self.url = vectordb_config.base_url
        self.api_key = vectordb_config.api_key
        self.api_version = vectordb_config.api_version
        self.index = vectordb_config.index
