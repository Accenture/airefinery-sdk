"""FastGraphRAG knowledge-graph module, supports building, updating, and querying
a knowledge graph
"""

from graphrag.config.enums import IndexingMethod

from air.knowledge.knowledge_graph.base_knowledge_graph import (
    BaseKnowledgeGraph,
)
from air.knowledge.knowledge_graph.graphrag import GraphRAG
from air.types import KnowledgeGraphConfig


class FastGraphRAG(BaseKnowledgeGraph):
    """
    FastGraphRAG knowledge graph class, inherits from BaseKnowledgeGraph
    """

    def __init__(
        self, config: KnowledgeGraphConfig
    ):  # pylint:disable=super-init-not-called
        """
        Initialize the GraphRAG module
        """

        self.graph_rag = GraphRAG(config=config)
        self.graph_rag.index_method = IndexingMethod.Fast

    async def build(self) -> bool:
        """
        Build the knowledge graph using the specified model.
        """
        return await self.graph_rag.build()

    async def update(self) -> bool:
        """
        Update the knowledge graph using the specified model.
        """
        return await self.graph_rag.update()

    async def query(self, query: str, method: str):
        """
        Query the knowledge graph using the specified query string.
        """
        # Implementation for querying the knowledge graph
        return await self.graph_rag.query(query=query, method=method)
