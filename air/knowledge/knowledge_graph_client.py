"""
Graph processing client. The semantic layer of Knowledge.
"""

import asyncio
import logging
import os
from typing import List, Union

import aiofiles

from air import __base_url__
from air.knowledge.graph_visualization import GraphProcessing
from air.knowledge.knowledge_graph import KnowledgeGraphRegistry
from air.types import Document, KnowledgeGraphConfig
from air.utils import copy_files, secure_join

logger = logging.getLogger(__name__)


class KnowledgeGraphClient:
    """
    Interface for interacting with the AI Refinery's knowledge extraction service,
    with knowledge represented as a graph.
    """

    def create_project(self, graph_config: KnowledgeGraphConfig):
        """
        Initializes and sets up a knowledge graph project based on the provided configuration.

        Args:
            graph_config (KnowledgeGraphConfig):  Configuration for the knowledge graph,
        including the type and working directory.

        Raises:
            ValueError: If the specified knowledge graph type is not registered.
        """
        for log_name, log_obj in logging.Logger.manager.loggerDict.items():
            if (
                log_name.startswith("graphrag")
                or "openai" in log_name
                or "httpx" in log_name
            ):
                log_obj.disabled = True  # type: ignore
        self.work_dir = graph_config.work_dir
        knowledge_graph_class = KnowledgeGraphRegistry.get(graph_config.type)
        if not knowledge_graph_class:
            raise ValueError(
                f"Knowledge Graph type {graph_config.type} is not available. Available Knowledge Graph types: {KnowledgeGraphRegistry.all_knowledgegraphs()}"
            )
        os.makedirs(self.work_dir, exist_ok=True)
        self.knowledge_graph = knowledge_graph_class(config=graph_config)

    async def build(
        self,
        files_path: str | None = None,
        docs: List[Document] | None = None,
    ) -> bool:
        """
        Build a graph from the provided data.
        Args:
            files_path (str): Path to the files to be processed.
            docs (List[Document]): List of document elements to be processed.
        Returns:
            bool: True if the graph was built successfully, False otherwise.
        """
        if os.path.exists(secure_join(self.work_dir, "output", "graph.graphml")):
            logger.error(
                "Knowledge Graph already exists! Call `update()` if you want to add more data to the knowledge graph."
            )
            return False

        input_folder = secure_join(self.work_dir, "input")
        os.makedirs(input_folder, exist_ok=True)
        if not files_path and not docs:
            raise ValueError("Either files_path or docs must be provided.")
        if files_path:
            status = await asyncio.to_thread(copy_files, files_path, input_folder)
            if not status:
                logger.error("Failed to copy files.")
                return False
        else:
            for doc in docs:  # type: ignore
                texts = [
                    element.text
                    for element in doc.elements
                    if element.element_type == "text"
                ]
                text_str = "\n".join(texts)
                async with aiofiles.open(
                    secure_join(input_folder, f"{doc.filename}.txt"),
                    "w",
                    encoding="utf-8",
                ) as fp:
                    await fp.write(text_str)
        logger.info("Running knowledge-graph build...")
        status = await self.knowledge_graph.build()
        return status

    async def query(
        self,
        query: str,
        method: str = "local",
    ) -> Union[str, None]:
        """
        Query the knowledge graph using the specified query string.
        Args:
            query (str): The query string to search for.
            method (str): The method to use for querying. Defaults to "local".
        Returns:
            str: The response generated by the LLM.
        """
        try:
            logger.info("Running query...")
            response = await self.knowledge_graph.query(
                query=query,
                method=method,
            )
        except ValueError:
            logger.error("Query failed, invalid search method.")
            return None
        return response

    async def update(
        self, files_path: str | None = None, docs: List[Document] | None = None
    ) -> bool:
        """
        Update the knowledge graph with new data
        Args:
            files_path (str): Path to the files to be processed.
            docs (List[Document]): List of document elements to be processed.
        Returns:
            bool: True if the graph was updated successfully, False otherwise.
        """
        input_folder = secure_join(self.work_dir, "input")
        os.makedirs(input_folder, exist_ok=True)
        if not files_path and not docs:
            raise ValueError("Either files_path or docs must be provided.")
        if files_path:
            status = await asyncio.to_thread(copy_files, files_path, input_folder)
            if not status:
                logger.error("Failed to copy files.")
                return False
        else:
            for doc in docs:  # type: ignore
                texts = [
                    element.text
                    for element in doc.elements
                    if element.element_type == "text"
                ]
                text_str = "\n".join(texts)
                async with aiofiles.open(
                    secure_join(input_folder, f"{doc.filename}.txt"),
                    "w",
                    encoding="utf-8",
                ) as fp:
                    await fp.write(text_str)
        logger.info("Running knowledge-graph update...")
        status = await self.knowledge_graph.update()
        return status

    def visualize(
        self,
        max_community_size: Union[int, None] = None,
        community_level: Union[int, None] = None,
        figsize: tuple[float, float] = (36.0, 20.0),
        default_node_sizes: int = 500,
        fig_format: str = "svg",
        dpi: int = 300,
        font_size: int = 10,
        scale_factor: int = 20,
    ) -> bool:
        """
        Function to visualize the graph and generate an SVG image of the graph.
        Uses the graph.graphml file under work_dir/output folder.
        Set optional parameters to filter the graph before visualizing.

        Parameters:
        max_community_size (Union[int, None], optional): maximum number of nodes to be present
        in a cluster/community. If set as None, clustering is skipped. Defaults to None.
        community_level (Union[int, None], optional): Level of the community to retain.
        Nodes of this community level are retained and then visualized.
        If set to `-1` highest community level nodes are retained. Defaults to None.

        Returns:
        - bool: True if processing and saving were successful, False otherwise.
        """
        graph_path = secure_join(self.work_dir, "output", "graph.graphml")
        graph_image_path = secure_join(self.work_dir, "output", "graph.svg")
        if not os.path.exists(graph_path):
            logger.error("GraphML file not found.")
            return False
        status = GraphProcessing.visualize_graph(
            graph_path=graph_path,
            graph_save_path=graph_image_path,
            max_community_size=max_community_size,
            community_level=community_level,
            figsize=figsize,
            default_node_sizes=default_node_sizes,
            fig_format=fig_format,
            dpi=dpi,
            font_size=font_size,
            scale_factor=scale_factor,
        )
        if status is False:
            logger.error("Graph visualization failed.")
            return False
        logger.info("Graph visualization completed successfully.")

        return True
