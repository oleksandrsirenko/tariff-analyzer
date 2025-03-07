"""Network analysis for tariff relationships."""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, Counter

from ..utils import logger


class TariffNetworkAnalyzer:
    """
    Analyzes tariff data as a network of trade relationships.
    """

    def __init__(self, data: Union[pd.DataFrame, Dict[str, Any]]):
        """
        Initialize with either a DataFrame or processed tariff data dictionary.

        Args:
            data: Processed tariff data as DataFrame or dictionary
        """
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, dict) and "events" in data:
            # Convert events to DataFrame
            self.df = self._convert_events_to_df(data["events"])
        else:
            raise ValueError(
                "Data must be a DataFrame or a dictionary with 'events' key"
            )

        # Initialize graph
        self._build_graph()

    def _convert_events_to_df(self, events: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert events to DataFrame.

        Args:
            events: List of event dictionaries

        Returns:
            DataFrame with flattened event data
        """
        flattened = []

        for event in events:
            tariff = event.get("tariffs_v2", {})

            # Extract relationship data
            imposing_country = tariff.get("imposing_country_code")
            targeted_countries = tariff.get("targeted_country_codes", [])

            if imposing_country and targeted_countries:
                for targeted_country in targeted_countries:
                    # Create one row per relationship
                    record = {
                        "event_id": event.get("id"),
                        "imposing_country": imposing_country,
                        "targeted_country": targeted_country,
                        "measure_type": tariff.get("measure_type"),
                        "main_tariff_rate": tariff.get("main_tariff_rate"),
                        "announcement_date_std": tariff.get("announcement_date_std"),
                        "impact_score": tariff.get("impact_score"),
                        "is_retaliatory": tariff.get("is_retaliatory", False),
                    }

                    flattened.append(record)

        return pd.DataFrame(flattened)

    def _build_graph(self) -> None:
        """Build directed graph representation of tariff relationships."""
        # Create directed graph
        self.graph = nx.DiGraph()

        # Count relationships
        relationship_counts = defaultdict(int)
        relationship_weight = defaultdict(float)

        # Add nodes and edges
        for _, row in self.df.iterrows():
            source = row.get("imposing_country")
            target = row.get("targeted_country")

            if not source or not target:
                continue

            # Add nodes
            if source not in self.graph:
                self.graph.add_node(source, type="country")

            if target not in self.graph:
                self.graph.add_node(target, type="country")

            # Track relationship
            relationship = (source, target)
            relationship_counts[relationship] += 1

            # Add impact to weight if available
            impact = row.get("impact_score") or row.get("main_tariff_rate") or 1.0
            relationship_weight[relationship] += float(impact)

        # Add weighted edges
        for (source, target), count in relationship_counts.items():
            weight = relationship_weight[(source, target)]
            self.graph.add_edge(source, target, weight=weight, count=count)

        logger.info(
            f"Built graph with {self.graph.number_of_nodes()} nodes and "
            f"{self.graph.number_of_edges()} edges"
        )

    def get_network_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the tariff network.

        Returns:
            Dictionary with network summary
        """
        if not self.graph:
            return {"error": "Graph not initialized"}

        # Basic graph metrics
        summary = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
        }

        # Degree statistics
        in_degrees = [d for _, d in self.graph.in_degree()]
        out_degrees = [d for _, d in self.graph.out_degree()]

        summary["degree_stats"] = {
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "avg_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
            "avg_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0,
        }

        # Centrality
        try:
            # Top countries by betweenness centrality
            betweenness = nx.betweenness_centrality(self.graph, weight="weight")
            top_betweenness = sorted(
                betweenness.items(), key=lambda x: x[1], reverse=True
            )[:5]

            # Top countries by in-degree centrality (most targeted)
            in_degree_centrality = nx.in_degree_centrality(self.graph)
            top_targeted = sorted(
                in_degree_centrality.items(), key=lambda x: x[1], reverse=True
            )[:5]

            # Top countries by out-degree centrality (most imposing)
            out_degree_centrality = nx.out_degree_centrality(self.graph)
            top_imposing = sorted(
                out_degree_centrality.items(), key=lambda x: x[1], reverse=True
            )[:5]

            summary["centrality"] = {
                "top_betweenness": [
                    {"country": c, "score": round(s, 3)} for c, s in top_betweenness
                ],
                "top_targeted": [
                    {"country": c, "score": round(s, 3)} for c, s in top_targeted
                ],
                "top_imposing": [
                    {"country": c, "score": round(s, 3)} for c, s in top_imposing
                ],
            }
        except Exception as e:
            logger.warning(f"Error calculating centrality measures: {e}")

        # Community detection (if network is large enough)
        if self.graph.number_of_nodes() >= 4:
            try:
                # Use undirected version for community detection
                undirected_graph = self.graph.to_undirected()
                communities = list(
                    nx.community.greedy_modularity_communities(undirected_graph)
                )

                community_data = []
                for i, comm in enumerate(communities):
                    community_data.append(
                        {"id": i, "size": len(comm), "countries": sorted(list(comm))}
                    )

                summary["communities"] = community_data
            except Exception as e:
                logger.warning(f"Error detecting communities: {e}")

        return summary

    def get_relationship_analysis(self) -> Dict[str, Any]:
        """
        Analyze the strength and patterns of relationships between countries.

        Returns:
            Dictionary with relationship analysis
        """
        if self.df.empty:
            return {"error": "No relationship data available"}

        # Count events between each pair of countries
        relationship_counts = self.df.groupby(
            ["imposing_country", "targeted_country"]
        ).size()

        # Convert to list format
        relationships = []
        for (source, target), count in relationship_counts.items():
            relationship = {"source": source, "target": target, "count": int(count)}

            # Add average impact if available
            impact_col = (
                "impact_score"
                if "impact_score" in self.df.columns
                else "main_tariff_rate"
            )
            if impact_col in self.df.columns:
                impact = self.df[
                    (self.df["imposing_country"] == source)
                    & (self.df["targeted_country"] == target)
                ][impact_col].mean()

                if not pd.isna(impact):
                    relationship["avg_impact"] = float(impact)

            relationships.append(relationship)

        # Sort by count
        relationships.sort(key=lambda x: x["count"], reverse=True)

        # Analyze reciprocal relationships
        reciprocal_pairs = set()
        reciprocal_relationships = []

        for i, rel1 in enumerate(relationships):
            source1, target1 = rel1["source"], rel1["target"]

            # Check if the pair has already been processed
            if (source1, target1) in reciprocal_pairs or (
                target1,
                source1,
            ) in reciprocal_pairs:
                continue

            # Look for reciprocal relationship
            for rel2 in relationships:
                source2, target2 = rel2["source"], rel2["target"]

                if source1 == target2 and target1 == source2:
                    # Found reciprocal pair
                    reciprocal_pairs.add((source1, target1))
                    reciprocal_pairs.add((target1, source1))

                    reciprocal_relationships.append(
                        {
                            "countries": [source1, target1],
                            "a_to_b_count": rel1["count"],
                            "b_to_a_count": rel2["count"],
                            "ratio": (
                                round(rel1["count"] / rel2["count"], 2)
                                if rel2["count"] > 0
                                else float("inf")
                            ),
                        }
                    )

                    break

        # Sort reciprocal relationships by total count
        reciprocal_relationships.sort(
            key=lambda x: x["a_to_b_count"] + x["b_to_a_count"], reverse=True
        )

        return {
            "top_relationships": relationships[:10],
            "reciprocal_relationships": reciprocal_relationships[:10],
            "total_relationships": len(relationships),
            "total_reciprocal": len(reciprocal_relationships),
        }

    def get_community_relations(self) -> Dict[str, Any]:
        """
        Analyze relations between detected communities.

        Returns:
            Dictionary with community relation data
        """
        if not self.graph or self.graph.number_of_nodes() < 4:
            return {"error": "Graph too small for community analysis"}

        try:
            # Use undirected version for community detection
            undirected_graph = self.graph.to_undirected()
            communities = list(
                nx.community.greedy_modularity_communities(undirected_graph)
            )

            # Map nodes to communities
            node_community = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_community[node] = i

            # Count relations between communities
            community_relations = defaultdict(int)
            for source, target, data in self.graph.edges(data=True):
                if source in node_community and target in node_community:
                    source_comm = node_community[source]
                    target_comm = node_community[target]

                    if source_comm != target_comm:
                        relation = (source_comm, target_comm)
                        community_relations[relation] += data.get("count", 1)

            # Format results
            relations = []
            for (source_comm, target_comm), count in community_relations.items():
                relations.append(
                    {
                        "source_community": source_comm,
                        "target_community": target_comm,
                        "count": count,
                    }
                )

            relations.sort(key=lambda x: x["count"], reverse=True)

            return {
                "communities": [sorted(list(comm)) for comm in communities],
                "relations": relations,
            }
        except Exception as e:
            logger.warning(f"Error analyzing community relations: {e}")
            return {"error": str(e)}

    def detect_retaliation_patterns(self) -> Dict[str, Any]:
        """
        Detect patterns of retaliatory tariffs between countries.

        Returns:
            Dictionary with retaliation pattern data
        """
        if "is_retaliatory" not in self.df.columns:
            return {"error": "No retaliation data available"}

        # Filter retaliatory events
        retaliatory_df = self.df[self.df["is_retaliatory"] == True]

        if retaliatory_df.empty:
            return {"error": "No retaliatory events found"}

        # Count retaliatory events by country pair
        retaliation_counts = retaliatory_df.groupby(
            ["imposing_country", "targeted_country"]
        ).size()

        # Convert to list format
        retaliations = []
        for (source, target), count in retaliation_counts.items():
            retaliations.append(
                {"retaliator": source, "target": target, "count": int(count)}
            )

        retaliations.sort(key=lambda x: x["count"], reverse=True)

        # Identify bidirectional retaliation (tit-for-tat)
        tit_for_tat = []
        processed_pairs = set()

        for ret1 in retaliations:
            source1, target1 = ret1["retaliator"], ret1["target"]
            pair = frozenset([source1, target1])

            if pair in processed_pairs:
                continue

            # Look for reciprocal retaliation
            for ret2 in retaliations:
                source2, target2 = ret2["retaliator"], ret2["target"]

                if source1 == target2 and target1 == source2:
                    # Found tit-for-tat
                    processed_pairs.add(pair)

                    tit_for_tat.append(
                        {
                            "countries": [source1, target1],
                            "a_to_b_count": ret1["count"],
                            "b_to_a_count": ret2["count"],
                        }
                    )

                    break

        # Count retaliatory events by country
        retaliation_by_country = defaultdict(lambda: {"imposed": 0, "received": 0})

        for _, row in retaliatory_df.iterrows():
            source = row["imposing_country"]
            target = row["targeted_country"]

            retaliation_by_country[source]["imposed"] += 1
            retaliation_by_country[target]["received"] += 1

        # Format country data
        country_data = []
        for country, counts in retaliation_by_country.items():
            country_data.append(
                {
                    "country": country,
                    "retaliation_imposed": counts["imposed"],
                    "retaliation_received": counts["received"],
                    "net_retaliation": counts["imposed"] - counts["received"],
                }
            )

        country_data.sort(key=lambda x: abs(x["net_retaliation"]), reverse=True)

        return {
            "retaliations": retaliations,
            "tit_for_tat": tit_for_tat,
            "country_retaliation": country_data,
        }

    def get_network_graph_data(self, include_weights: bool = True) -> Dict[str, Any]:
        """
        Get graph data for visualization.

        Args:
            include_weights: Whether to include edge weights

        Returns:
            Dictionary with nodes and edges for graph visualization
        """
        if not self.graph:
            return {"error": "Graph not initialized"}

        # Collect node data
        nodes = []
        for node, data in self.graph.nodes(data=True):
            node_data = {"id": node, "type": data.get("type", "country")}

            # Add degree information
            node_data["in_degree"] = self.graph.in_degree(node)
            node_data["out_degree"] = self.graph.out_degree(node)

            nodes.append(node_data)

        # Collect edge data
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edge_data = {
                "source": source,
                "target": target,
                "count": data.get("count", 1),
            }

            if include_weights:
                edge_data["weight"] = data.get("weight", 1.0)

            edges.append(edge_data)

        return {"nodes": nodes, "edges": edges}
