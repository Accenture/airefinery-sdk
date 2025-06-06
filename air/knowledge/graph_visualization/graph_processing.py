import logging
from typing import Union

import networkx as nx
import pandas as pd
from graphrag.index.operations.cluster_graph import cluster_graph as graphrag_clustering

from air.knowledge.graph_visualization.graph_display import GraphDisplay

logger = logging.getLogger(__name__)


class GraphProcessing:
    """
    Class that performs graph clustering and visualization
    """

    @classmethod
    def cluster_graph(
        cls, graph: nx.Graph, max_community_size: int = 1
    ) -> pd.DataFrame:
        """
        Method to perform hierarchical clustering of given graph,
        until the resulting final communities each have a maximum of
        `max_community_size` number of nodes.

        Args:
            graph (nx.Graph): graph to be clustered
            max_community_size (int, optional): maximum number of nodes to be present
            in a cluster/community. Defaults to 1.

        Returns:
            pd.DataFrame: Clustered communities in a DataFrame format.
            Columns are - level, cluster_id, parent_cluster_id, nodes
        """

        cluster_communities = graphrag_clustering(
            graph=graph, max_cluster_size=max_community_size, use_lcc=False, seed=42
        )
        cluster_communities = pd.DataFrame(
            cluster_communities,
            columns=[
                "level",
                "cluster_id",
                "parent_cluster_id",
                "nodes",
            ],  # type: ignore
        )
        return cluster_communities

    @classmethod
    def add_node_labels(cls, cluster_communities: pd.DataFrame, graph: nx.Graph):
        """
        Method to add community name and community level
        to the graph nodes

        Args:
            cluster_communities (pd.DataFrame): cluster info dataframe
            graph (nx.Graph): networkx graph

        Returns:
            nx.Graph: updated graph with community and community level labels
        """
        cluster_communities.set_index("cluster_id", inplace=True)
        node_comm_map = {
            node: community
            for community, community_info in cluster_communities.iterrows()
            for node in community_info["nodes"]
        }
        for node in graph.nodes:
            if node not in node_comm_map:
                continue
            community = node_comm_map[node]
            graph.nodes[node]["community"] = str(community)
            graph.nodes[node]["community_level"] = int(
                cluster_communities.at[community, "level"]
            )
        return graph

    @classmethod
    def visualize_graph(
        cls,
        graph_path: str,
        graph_save_path: str,
        max_community_size: Union[int, None] = None,
        community_level: Union[int, None] = None,
        figsize: tuple[float, float] = (36.0, 20.0),
        default_node_sizes: int = 500,
        fig_format: str = "svg",
        dpi: int = 300,
        font_size: int = 10,
        scale_factor: int = 20,
    ):
        """
        Method to perform graph clustering, if specified,
        and visualize the resulting graph using the GraphDisplay class.

        Args:
            graph_path (str): path to the graphml file
            graph_save_path (str): path where the resulting graph visualization is to be saved
            max_community_size (Union[int, None], optional): maximum number of nodes to be present
            in a cluster/community. If set as None, clustering is skipped. Defaults to None.
            community_level (Union[int, None], optional): Level of the community to retain.
            Nodes of this community level are retained and then visualized.
            If set to `-1` highest community level nodes are retained. Defaults to None.
        """
        status = False
        try:
            graph = nx.read_graphml(graph_path)
            filtered_graph = graph
            if max_community_size:
                cluster_communities_df = cls.cluster_graph(
                    graph=graph, max_community_size=max_community_size
                )
                graph = cls.add_node_labels(cluster_communities_df, graph)
            if community_level is not None:
                community_levels = nx.get_node_attributes(graph, "community_level")
                if not community_levels:
                    logger.warning(
                        "Community level labels not present in graphml. Skipping community level filter..."
                    )
                else:
                    max_community_level = max(community_levels.values())
                    if community_level > max_community_level:
                        logger.warning(
                            "Community level specified is greater than the max community level in the graph. Skipping community level filter..."
                        )
                    else:
                        community_level = (
                            max_community_level
                            if community_level == -1
                            else community_level
                        )
                        filtered_nodes = [
                            node
                            for node, comm_level in community_levels.items()
                            if int(comm_level) == community_level
                        ]
                        filtered_graph = graph.subgraph(filtered_nodes)
            status = GraphDisplay.show_undirected_graph(
                filtered_graph,
                graph_save_path,
                figsize,
                default_node_sizes,
                fig_format,
                dpi,
                font_size,
                scale_factor,
            )
        except Exception as e:
            logger.error("An error occurred during graph visualization %s", str(e))
        return status
