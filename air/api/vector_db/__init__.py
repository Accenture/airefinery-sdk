"""
Module for vector databases
"""

from air.api.vector_db.azure_aisearch import AzureAISearch
from air.api.vector_db.vectordb_registry import VectorDBRegistry
from air.api.vector_db.base_vectordb import BaseVectorDB, VectorDBConfig

__all__ = ["VectorDBRegistry", "BaseVectorDB", "VectorDBConfig"]
