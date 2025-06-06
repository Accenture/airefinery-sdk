"""
GraphRAG knowledge-graph module, supports building, updating, and querying
a knowledge graph
"""

import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Union

import graphrag.api as api
import networkx as nx
import pandas as pd
from graphrag.cli.initialize import initialize_project_at
from graphrag.cli.query import _resolve_output_files
from graphrag.config.enums import IndexingMethod, SearchMethod
from graphrag.config.load_config import load_config
from graphrag.logger.factory import LoggerFactory

from air.knowledge.knowledge_graph.base_knowledge_graph import (
    BaseKnowledgeGraph,
)
from air.types import KnowledgeGraphConfig
from air.utils import secure_join

logger = logging.getLogger(__name__)


class HiddenPrints:
    """
    Class to suppress print statements
    """

    def __enter__(self):
        # pylint: disable=attribute-defined-outside-init
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class GraphRAG(BaseKnowledgeGraph):
    """
    GraphRAG knowledge graph class, inherits from BaseKnowledgeGraph
    """

    def __init__(self, config: KnowledgeGraphConfig):
        """
        Initialize the GraphRAG module
        """

        super().__init__(graph_config=config)
        self.index_method = IndexingMethod.Standard
        self.progress_logger = LoggerFactory().create_logger("none")

        # secure path validation for settings.yaml
        settings_path = secure_join(self.work_dir, "settings.yaml")
        if not os.path.exists(settings_path):
            with HiddenPrints():
                initialize_project_at(
                    path=Path(self.work_dir),
                    force=False,
                )
        self.config_overrides = {
            "models.default_chat_model.api_base": self.base_url,
            "models.default_chat_model.api_key": self.api_key,
            "models.default_embedding_model.api_base": self.base_url,
            "models.default_embedding_model.api_key": self.api_key,
            "chunks.size": self.chunk_size,
            "chunks.overlap": self.chunk_overlap,
            "snapshots.graphml": True,
        }
        if self.api_type == "azure":
            self.config_overrides.update(
                {
                    "models.default_chat_model.type": "azure_openai_chat",
                    "models.default_chat_model.deployment_name": self.llm_model,
                    "models.default_chat_model.api_version": "2024-05-01-preview",
                    "models.default_embedding_model.type": "azure_openai_embedding",
                    "models.default_embedding_model.deployment_name": self.embedding_model,
                    "models.default_embedding_model.api_version": "2023-05-15",
                }
            )
        else:
            self.config_overrides.update(
                {
                    "models.default_chat_model.model": self.llm_model,
                    "models.default_embedding_model.model": self.embedding_model,
                }
            )
        self.graph_config = load_config(
            root_dir=Path(self.work_dir), cli_overrides=self.config_overrides
        )

    @staticmethod
    def add_node_edge_labels(
        communities_df: pd.DataFrame,
        entities_df: pd.DataFrame,
        relationships_df: pd.DataFrame,
        graph: nx.Graph,
    ):
        """
        Function to add community labels to the nodes in the graphml file
        """

        communities_df.set_index("community", inplace=True)
        ent_com_map = {
            entity: community
            for community, community_info in communities_df.iterrows()
            for entity in community_info["entity_ids"]
        }
        ent_com_map["default"] = "unnamed"
        node_ent_map = {
            entity["title"]: entity["id"] for _, entity in entities_df.iterrows()
        }
        for node in graph.nodes:
            community = ent_com_map.get(
                node_ent_map.get(str(node), "default"), "unnamed"
            )
            graph.nodes[node]["community"] = str(community)
            if community not in communities_df.index:
                graph.nodes[node]["community_level"] = int(0)
            else:
                graph.nodes[node]["community_level"] = int(
                    communities_df.at[community, "level"]
                )
        relationships_df.set_index(["source", "target"], inplace=True)
        for edge in graph.edges:
            if edge in relationships_df.index:
                graph.edges[edge]["weight"] = float(relationships_df.at[edge, "weight"])
            else:
                graph.edges[edge]["weight"] = float(1.0)
        return graph

    async def build(self) -> bool:
        """
        Build the knowledge graph using the specified model.
        """

        try:
            with HiddenPrints():
                build_index_result = await api.build_index(
                    config=self.graph_config,
                    method=self.index_method,
                    progress_logger=None,
                )
        except Exception as e:
            logger.error("Exception occurred during knowledge graph build phase!")
            logger.error(e)
            return False
        error = False
        for result in build_index_result:
            if result.errors:
                logger.error("Error occurred during %s phase", result.workflow)
                for workflow_error in result.errors:
                    logger.error("%s", workflow_error)
                error = True
        if error:
            return False
        try:
            # using secure path handling for all file operations
            output_dir = secure_join(self.work_dir, "output")
            communities_path = secure_join(output_dir, "communities.parquet")
            entities_path = secure_join(output_dir, "entities.parquet")
            relationships_path = secure_join(output_dir, "relationships.parquet")
            graph_path = secure_join(output_dir, "graph.graphml")

            communities_df = await asyncio.to_thread(pd.read_parquet, communities_path)
            entities_df = await asyncio.to_thread(pd.read_parquet, entities_path)
            relationships_df = await asyncio.to_thread(
                pd.read_parquet, relationships_path
            )
            graph = await asyncio.to_thread(nx.read_graphml, graph_path)
            graph = self.add_node_edge_labels(
                communities_df,
                entities_df,
                relationships_df,
                graph,
            )
            await asyncio.to_thread(nx.write_graphml, graph, graph_path)
        except Exception as e:
            logger.error(
                "Exception occurred while adding community labels to the graphml file!"
            )
            logger.error(e)
        return True

    async def update(self) -> bool:
        """
        Update the knowledge graph using the specified model.
        """
        try:
            update_dir = secure_join(self.work_dir, "update_output")
            original_graph_path = secure_join(self.work_dir, "output", "graph.graphml")
        except ValueError:
            logger.error("Parse traversal detected during knowledge graph update!")
            return False
        original_graph = nx.Graph()
        if os.path.exists(original_graph_path):
            original_graph = await asyncio.to_thread(
                nx.read_graphml, original_graph_path
            )
        else:
            logger.error(
                "Original graph.graphml file is missing!!! Cannot run `update`"
            )
            return False
        try:
            with HiddenPrints():
                update_index_result = await api.build_index(
                    config=self.graph_config,
                    method=self.index_method,
                    is_update_run=True,
                    progress_logger=None,
                )
        except Exception as e:
            logger.error("Exception occurred during knowledge graph update phase!")
            logger.error(e)
            logger.info("Removing directory: %s", update_dir)
            await asyncio.to_thread(shutil.rmtree, update_dir, ignore_errors=True)
            return False
        error = False
        for result in update_index_result:
            if result.errors:
                logger.error("Error occurred during %s phase", result.workflow)
                for workflow_error in result.errors:
                    logger.error("%s", workflow_error)
                error = True
        if error:
            return False
        updated_graph = nx.Graph()
        if not os.path.exists(update_dir):
            return True

        update_dirs = os.listdir(update_dir)
        if not update_dirs:
            return True

        # enhanced validation of the subdirectory name to prevent path traversal
        subdirectory = update_dirs[0]
        try:
            updated_graph_path = secure_join(
                update_dir, subdirectory, "delta", "graph.graphml"
            )
        except ValueError:
            logger.error(
                "Invalid subdirectory name detected: %s during knowledge graph update!",
                subdirectory,
            )
            return False
        updated_graph = await asyncio.to_thread(nx.read_graphml, updated_graph_path)
        updated_graph = nx.compose(original_graph, updated_graph)  # type:ignore
        try:
            # use secure path handling for all file operations
            output_dir = secure_join(self.work_dir, "output")
            communities_path = secure_join(output_dir, "communities.parquet")
            entities_path = secure_join(output_dir, "entities.parquet")
            relationships_path = secure_join(output_dir, "relationships.parquet")

            communities_df = await asyncio.to_thread(pd.read_parquet, communities_path)
            entities_df = await asyncio.to_thread(pd.read_parquet, entities_path)
            relationships_df = await asyncio.to_thread(
                pd.read_parquet, relationships_path
            )
            updated_graph = self.add_node_edge_labels(
                communities_df,
                entities_df,
                relationships_df,
                updated_graph,
            )
        except Exception as e:
            logger.error(
                "Exception occurred while adding community labels to the graphml file!"
            )
            logger.error(e)
        await asyncio.to_thread(nx.write_graphml, updated_graph, original_graph_path)
        logger.info("Removing directory: %s", update_dir)
        await asyncio.to_thread(shutil.rmtree, update_dir, ignore_errors=True)
        return True

    async def query(self, query: str, method: str) -> Union[str, None]:
        """
        Query the knowledge graph using the specified query string.
        """

        try:
            search_method = SearchMethod(method)
        except ValueError as exc:
            logger.error(
                "Invalid Query Search Method: %s. Available methods: local, global, drift, basic",
                method,
            )
            raise ValueError(
                f"Invalid Query Search Method: {method}. Available methods: local, global, drift, basic"
            ) from exc
        response = None
        try:
            dataframe_dict = await asyncio.to_thread(
                _resolve_output_files,
                self.graph_config,
                [
                    "communities",
                    "community_reports",
                    "text_units",
                    "relationships",
                    "entities",
                ],
                [
                    "covariates",
                ],
            )
            final_communities: pd.DataFrame = dataframe_dict["communities"]
            final_community_reports: pd.DataFrame = dataframe_dict["community_reports"]
            final_text_units: pd.DataFrame = dataframe_dict["text_units"]
            final_relationships: pd.DataFrame = dataframe_dict["relationships"]
            final_entities: pd.DataFrame = dataframe_dict["entities"]
            final_covariates: pd.DataFrame | None = dataframe_dict["covariates"]
            match search_method:
                case SearchMethod.LOCAL:
                    with HiddenPrints():
                        response, _ = await api.local_search(
                            config=self.graph_config,
                            entities=final_entities,
                            communities=final_communities,
                            community_reports=final_community_reports,
                            text_units=final_text_units,
                            relationships=final_relationships,
                            covariates=final_covariates,
                            community_level=2,
                            response_type="Multiple Paragraphs",
                            query=query,
                        )
                case SearchMethod.GLOBAL:
                    with HiddenPrints():
                        response, _ = await api.global_search(
                            config=self.graph_config,
                            entities=final_entities,
                            communities=final_communities,
                            community_reports=final_community_reports,
                            community_level=2,
                            dynamic_community_selection=False,
                            response_type="Multiple Paragraphs",
                            query=query,
                        )
                case SearchMethod.DRIFT:
                    with HiddenPrints():
                        response, _ = await api.drift_search(
                            config=self.graph_config,
                            entities=final_entities,
                            communities=final_communities,
                            community_reports=final_community_reports,
                            text_units=final_text_units,
                            relationships=final_relationships,
                            community_level=2,
                            response_type="Multiple Paragraphs",
                            query=query,
                        )
                case SearchMethod.BASIC:
                    with HiddenPrints():
                        response, _ = await api.basic_search(
                            config=self.graph_config,
                            text_units=final_text_units,
                            query=query,
                        )
        except Exception as e:
            logger.error("Exception occurred during querying")
            logger.error(e)
        return response  # type: ignore
