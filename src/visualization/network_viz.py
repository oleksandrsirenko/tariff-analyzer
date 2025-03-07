"""Network visualizations for tariff relationship data."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, Optional, Union, Tuple
import io
import base64

from ..utils import logger, config


class NetworkVisualizer:
    """
    Network visualizations for tariff relationship data.
    """

    def __init__(self, data: Union[pd.DataFrame, Dict[str, Any]]):
        """
        Initialize the network visualizer.

        Args:
            data: DataFrame or dictionary with tariff data
        """
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, dict) and "events" in data:
            # Convert events to DataFrame
            from ..preprocessing.processor import TariffProcessor

            processor = TariffProcessor()
            self.df = processor.to_dataframe(data)
        else:
            raise ValueError(
                "Data must be a DataFrame or a dictionary with 'events' key"
            )

        # Build network graph
        self.graph = self._build_graph()

        # Set style
        plt.style.use("seaborn-v0_8-whitegrid")

    def _build_graph(self) -> nx.DiGraph:
        """
        Build a directed graph from tariff data.

        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()

        # Check required columns
        if (
            "imposing_country_code" not in self.df.columns
            or "targeted_country_codes" not in self.df.columns
        ):
            logger.warning("Missing required columns for network graph")
            return G

        # Add edges for each tariff imposition
        for _, row in self.df.iterrows():
            source = row.get("imposing_country_code")

            # Handle targets as a list or semicolon-separated string
            targets = row.get("targeted_country_codes", "")
            if isinstance(targets, str):
                targets = [t.strip() for t in targets.split(";") if t.strip()]

            # Get measure type and rate (if available)
            measure_type = row.get("measure_type", "unknown")
            rate = row.get("main_tariff_rate", 0)
            is_retaliatory = row.get("is_retaliatory", False)

            for target in targets:
                if source and target:
                    # Add nodes if they don't exist
                    if source not in G:
                        G.add_node(source, count_imposing=0, count_targeted=0)
                    if target not in G:
                        G.add_node(target, count_imposing=0, count_targeted=0)

                    # Update node counts
                    G.nodes[source]["count_imposing"] = (
                        G.nodes[source].get("count_imposing", 0) + 1
                    )
                    G.nodes[target]["count_targeted"] = (
                        G.nodes[target].get("count_targeted", 0) + 1
                    )

                    # Add or update edge
                    if G.has_edge(source, target):
                        G[source][target]["weight"] += 1
                        G[source][target]["rates"].append(rate)
                        if is_retaliatory:
                            G[source][target]["retaliatory_count"] += 1
                        # Add measure type to the set of types
                        G[source][target]["measure_types"].add(measure_type)
                    else:
                        G.add_edge(
                            source,
                            target,
                            weight=1,
                            rates=[rate] if rate else [],
                            retaliatory_count=1 if is_retaliatory else 0,
                            measure_types={measure_type} if measure_type else set(),
                        )

        # Add more node attributes after all edges are processed
        for node in G.nodes():
            # Calculate ratio of imposing vs targeted
            imposing = G.nodes[node].get("count_imposing", 0)
            targeted = G.nodes[node].get("count_targeted", 0)

            # Prevent division by zero
            if targeted > 0:
                G.nodes[node]["imposing_targeted_ratio"] = imposing / targeted
            else:
                G.nodes[node]["imposing_targeted_ratio"] = (
                    float("inf") if imposing > 0 else 0
                )

            # Calculate node centrality
            G.nodes[node]["centrality"] = imposing + targeted

        logger.info(
            f"Built network graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )
        return G

    def plot_tariff_network(
        self, figsize: Tuple[int, int] = (12, 12), min_edge_weight: int = 1
    ) -> str:
        """
        Plot network graph of tariff relationships.

        Args:
            figsize: Figure size (width, height) in inches
            min_edge_weight: Minimum edge weight to include in the visualization

        Returns:
            Base64 encoded PNG image
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            return ""

        # Create a copy of the graph to filter
        G = self.graph.copy()

        # Filter edges by minimum weight
        edges_to_remove = [
            (u, v) for u, v, d in G.edges(data=True) if d["weight"] < min_edge_weight
        ]
        G.remove_edges_from(edges_to_remove)

        # Remove isolated nodes
        isolated_nodes = [n for n in G.nodes() if G.degree(n) == 0]
        G.remove_nodes_from(isolated_nodes)

        if G.number_of_nodes() == 0:
            return ""

        # Create figure
        plt.figure(figsize=figsize)

        # Get layout based on configuration
        layout_name = config.get("visualization.network_layout", "spring")

        if layout_name == "spring":
            pos = nx.spring_layout(G, seed=42, k=0.3)
        elif layout_name == "circular":
            pos = nx.circular_layout(G)
        elif layout_name == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout_name == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42, k=0.3)

        # Get node size based on centrality
        node_sizes = [max(300, G.nodes[n]["centrality"] * 100) for n in G.nodes()]

        # Determine node colors based on imposing/targeted ratio
        node_colors = []
        for n in G.nodes():
            ratio = G.nodes[n].get("imposing_targeted_ratio", 0)
            if ratio > 2:  # Mostly imposing
                node_colors.append("firebrick")
            elif ratio < 0.5:  # Mostly targeted
                node_colors.append("steelblue")
            else:  # Balanced
                node_colors.append("darkgreen")

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

        # Get edge weights and calculate widths
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1

        # Normalize edge widths based on weights
        edge_widths = [max(1, (w / max_weight) * 5) for w in edge_weights]

        # Create edge colors based on retaliatory percentage
        edge_colors = []
        for u, v in G.edges():
            retaliatory = G[u][v].get("retaliatory_count", 0)
            total = G[u][v].get("weight", 0)

            if total > 0 and retaliatory / total > 0.5:
                edge_colors.append("darkorange")  # Mostly retaliatory
            else:
                edge_colors.append("gray")  # Regular tariffs

        # Draw edges
        nx.draw_networkx_edges(
            G,
            pos,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.7,
            arrowsize=15,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos, font_size=10, font_weight="bold", font_color="black"
        )

        # Create legend elements
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="firebrick", edgecolor="black", label="Mainly Imposing"),
            Patch(facecolor="steelblue", edgecolor="black", label="Mainly Targeted"),
            Patch(facecolor="darkgreen", edgecolor="black", label="Balanced"),
            Line2D([0], [0], color="darkorange", lw=2, label="Retaliatory Tariffs"),
            Line2D([0], [0], color="gray", lw=2, label="Regular Tariffs"),
        ]

        plt.legend(handles=legend_elements, loc="upper right")

        plt.title("Tariff Relationship Network", fontsize=15)
        plt.axis("off")
        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def plot_community_network(self, figsize: Tuple[int, int] = (12, 12)) -> str:
        """
        Plot network with communities detected.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if not self.graph or self.graph.number_of_nodes() < 4:
            return ""

        # Create an undirected copy for community detection
        G_undirected = self.graph.to_undirected()

        # Detect communities
        try:
            communities = list(nx.community.greedy_modularity_communities(G_undirected))
        except Exception as e:
            logger.warning(f"Error detecting communities: {e}")
            return ""

        if not communities:
            return ""

        # Create a map from node to community
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i

        # Create figure
        plt.figure(figsize=figsize)

        # Get layout
        pos = nx.spring_layout(self.graph, seed=42, k=0.3)

        # Create a colormap
        import matplotlib.cm as cm

        num_communities = len(communities)
        community_colors = cm.viridis(np.linspace(0, 1, num_communities))

        # Draw nodes by community
        for i, community in enumerate(communities):
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=list(community),
                node_color=[community_colors[i]] * len(community),
                node_size=[300 + self.graph.degree(n) * 30 for n in community],
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
                label=f"Community {i+1}",
            )

        # Draw edges, thicker for intra-community
        for u, v in self.graph.edges():
            if community_map.get(u) == community_map.get(v):
                # Intra-community edge
                nx.draw_networkx_edges(
                    self.graph,
                    pos,
                    edgelist=[(u, v)],
                    width=2,
                    alpha=0.8,
                    edge_color="black",
                    arrowstyle="-|>",
                    arrowsize=15,
                )
            else:
                # Inter-community edge
                nx.draw_networkx_edges(
                    self.graph,
                    pos,
                    edgelist=[(u, v)],
                    width=1,
                    alpha=0.5,
                    edge_color="gray",
                    style="dashed",
                    arrowstyle="-|>",
                    arrowsize=10,
                )

        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos, font_size=10, font_weight="bold", font_color="black"
        )

        plt.title(
            f"Trade Communities (Modularity-Based, {num_communities} communities)",
            fontsize=15,
        )
        plt.axis("off")
        plt.legend(scatterpoints=1, loc="upper right")
        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def plot_centrality_ranking(
        self, figsize: Tuple[int, int] = (10, 8), top_n: int = 10
    ) -> str:
        """
        Plot ranking of countries by centrality measures.

        Args:
            figsize: Figure size (width, height) in inches
            top_n: Number of top countries to show

        Returns:
            Base64 encoded PNG image
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            return ""

        # Calculate centrality measures
        in_degree = nx.in_degree_centrality(self.graph)
        out_degree = nx.out_degree_centrality(self.graph)
        betweenness = nx.betweenness_centrality(self.graph)

        # Combine measures for each country
        countries = list(self.graph.nodes())
        measures = {
            "In-degree": [in_degree.get(c, 0) for c in countries],
            "Out-degree": [out_degree.get(c, 0) for c in countries],
            "Betweenness": [betweenness.get(c, 0) for c in countries],
        }

        # Create a DataFrame for easier manipulation
        df = pd.DataFrame(measures, index=countries)

        # Add descriptive labels for each measure
        labels = {
            "In-degree": "Targeted by others",
            "Out-degree": "Imposes on others",
            "Betweenness": "Trade relationship broker",
        }

        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=False)

        # Plot each measure
        for i, (measure, label) in enumerate(labels.items()):
            # Sort and select top N
            top = df.sort_values(by=measure, ascending=False).head(top_n)

            # Plot horizontal bar chart
            bars = axs[i].barh(top.index, top[measure], color="steelblue", alpha=0.7)

            # Add value labels
            for bar in bars:
                width = bar.get_width()
                axs[i].text(
                    width + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.3f}",
                    va="center",
                )

            # Set labels
            axs[i].set_title(f"Top Countries by {measure} Centrality")
            axs[i].set_xlabel("Centrality Score")
            axs[i].set_ylabel("Country")

            # Add explanatory text
            axs[i].text(
                0.95,
                0.05,
                label,
                transform=axs[i].transAxes,
                fontsize=9,
                style="italic",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.5"),
            )

        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def plot_reciprocal_relationships(self, figsize: Tuple[int, int] = (12, 6)) -> str:
        """
        Plot reciprocal tariff relationships.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if not self.graph or self.graph.number_of_edges() == 0:
            return ""

        # Find reciprocal relationships
        reciprocal = []
        for u, v in self.graph.edges():
            if self.graph.has_edge(v, u):
                # Only add each pair once
                if (v, u) not in reciprocal:
                    reciprocal.append((u, v))

        if not reciprocal:
            return ""

        # Calculate relationship stats
        recip_stats = []
        for u, v in reciprocal:
            u_to_v = self.graph[u][v]["weight"]
            v_to_u = self.graph[v][u]["weight"]

            u_to_v_retaliatory = self.graph[u][v].get("retaliatory_count", 0)
            v_to_u_retaliatory = self.graph[v][u].get("retaliatory_count", 0)

            ratio = (
                max(u_to_v, v_to_u) / min(u_to_v, v_to_u)
                if min(u_to_v, v_to_u) > 0
                else float("inf")
            )

            recip_stats.append(
                {
                    "pair": f"{u} ↔ {v}",
                    "u_to_v": u_to_v,
                    "v_to_u": v_to_u,
                    "total": u_to_v + v_to_u,
                    "ratio": ratio,
                    "retaliatory_percent": (
                        (u_to_v_retaliatory + v_to_u_retaliatory) / (u_to_v + v_to_u)
                        if (u_to_v + v_to_u) > 0
                        else 0
                    ),
                }
            )

        # Sort by total events
        recip_stats.sort(key=lambda x: x["total"], reverse=True)

        # Limit to top 10
        recip_stats = recip_stats[:10]

        # Create figure
        plt.figure(figsize=figsize)

        # Set up data for plotting
        pairs = [stat["pair"] for stat in recip_stats]
        u_to_v_values = [stat["u_to_v"] for stat in recip_stats]
        v_to_u_values = [stat["v_to_u"] for stat in recip_stats]

        # Set up x positions
        x = np.arange(len(pairs))
        width = 0.35

        # Create bars
        plt.barh(
            x - width / 2,
            u_to_v_values,
            width,
            color="steelblue",
            label="Left to Right",
        )
        plt.barh(
            x + width / 2,
            v_to_u_values,
            width,
            color="indianred",
            label="Right to Left",
        )

        # Set labels
        plt.yticks(x, pairs)
        plt.xlabel("Number of Tariff Events")
        plt.title("Reciprocal Tariff Relationships", fontsize=15)
        plt.legend()

        # Add grid
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Add retaliatory markers
        for i, stat in enumerate(recip_stats):
            if stat["retaliatory_percent"] > 0.5:
                plt.annotate(
                    "⚠ High retaliation",
                    xy=(max(stat["u_to_v"], stat["v_to_u"]) + 0.5, i),
                    va="center",
                    color="darkorange",
                    fontweight="bold",
                )

        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 encoded string.

        Args:
            fig: Matplotlib figure

        Returns:
            Base64 encoded PNG image
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        return img_str
