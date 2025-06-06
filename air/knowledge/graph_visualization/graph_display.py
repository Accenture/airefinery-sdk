"""
Module that show processed graph
"""

# pylint: disable=wrong-import-position
import logging
from typing import List, Union

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import colormaps as cm

logger = logging.getLogger(__name__)


class GraphDisplay:  # pylint: disable=too-few-public-methods
    """
    Base class that show processed graph
    """

    @classmethod
    def _map_edge_color(cls, graph: nx.Graph):
        """
        Map the graphnode weight to a color.

        Parameters:
        - graph (nxGraph): networkx graph

        Return:
        - List: The list of color code
        """

        edge_weights: List[Union[int, float]] = [
            data.get("weight", 1.0) for _, _, data in graph.edges(data=True)
        ]

        # Normalize the weights to a range [0, 1]
        weights_array = np.array(edge_weights, dtype=float)
        min_w = weights_array.min()
        max_w = weights_array.max()
        if max_w > min_w:
            norm_weights = (weights_array - min_w) / (max_w - min_w)
        else:
            norm_weights = (weights_array) / max_w

        # Choose a colormap from matplotlib (e.g., 'viridis')
        cmap = cm.get_cmap("viridis")

        # Map normalized weights to colors using the colormap
        edge_colors = [cmap(w) for w in norm_weights]

        return edge_colors

    @classmethod
    def show_undirected_graph(
        cls,
        graph,
        output_file: str,
        figsize: tuple[float, float] = (36.0, 20.0),
        default_node_sizes: int = 500,
        fig_format: str = "svg",
        dpi: int = 300,
        font_size: int = 10,
        scale_factor: int = 20,
    ) -> bool:
        """
        Reads a .graphml file and displays the undirected graph.

        Parameters:
        - graph (str): graph to be visualized, in networkx.Graph format
        - output_file (str): Path to the output graph image
        """

        try:
            node_sizes = {}
            node_colors = []
            node_labels = {}
            # Load the graph
            if graph.is_directed():
                graph = graph.to_undirected()

            # Extract community information
            communities = nx.get_node_attributes(graph, "community")
            if communities:
                # Assign colors to nodes based on their community
                unique_communities = set(communities.values())
                community_color_map = {
                    community: i for i, community in enumerate(unique_communities)
                }
                node_colors = [
                    community_color_map[communities[node]] for node in graph.nodes()
                ]
            node_labels = {str(node): str(node) for node in graph.nodes()}

            # Extract node sizes from the 'v_node_size' attribute
            node_sizes = nx.get_node_attributes(graph, "node_size")
            if not node_sizes:
                node_sizes = default_node_sizes
            else:
                node_sizes = [node_sizes[node] * scale_factor for node in graph.nodes()]

            if not node_colors:
                node_colors = "lightblue"

            edge_colors = cls._map_edge_color(graph)
            # Draw the graph with community colors
            plt.figure(figsize=figsize)
            pos = nx.spring_layout(
                graph, seed=42
            )  # Use a consistent layout for better visualization
            nx.draw(
                graph,
                pos,
                with_labels=True,
                labels=node_labels,
                node_color=node_colors,
                cmap=plt.get_cmap("rainbow"),
                node_size=node_sizes,
                font_size=font_size,
                edge_color=edge_colors,
            )
            plt.title("Graph with Communities")
            # Save or show the graph
            plt.savefig(
                output_file, format=fig_format, dpi=dpi
            )  # Save as a high-quality SVG file
            plt.clf()
            plt.close()
            logger.info("Graph saved to %s", output_file)

        except nx.NetworkXError as e:
            logger.error("Networkx graph file error %s", str(e))
            return False

        except Exception as e:  #   pylint: disable=broad-exception-caught
            logger.error("An error occurred during graph display: %s", str(e))
            return False
        return True
