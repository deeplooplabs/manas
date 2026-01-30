"""Graph visualization components for ManasRAG knowledge graphs.

This module provides interactive visualization capabilities for knowledge graphs,
community structures, query paths, and entity statistics using pyvis and plotly.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pyvis.network import Network

from manasrag.core.community import Community
from manasrag.stores.base import GraphDocumentStore
from manasrag.utils.color_utils import (
    create_colormap,
    generate_community_colors,
    get_gradient_color,
    get_type_color,
)


class GraphVisualizer:
    """Interactive visualization component for ManasRAG knowledge graphs.

    Generates HTML-based interactive visualizations using pyvis for network graphs
    and plotly for statistical charts.
    """

    def __init__(
        self,
        output_dir: str = "./manas_visualizations",
        default_layout: str = "hierarchical",
    ):
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save visualization files.
            default_layout: Default layout algorithm for network graphs.
        """
        self.output_dir = Path(output_dir)
        self.default_layout = default_layout

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_knowledge_graph(
        self,
        graph_store: GraphDocumentStore,
        output_path: Optional[str] = None,
        layout: str = "force",
        color_by: str = "entity_type",
        node_size: str = "degree",
        filter_min_degree: int = 0,
        filter_max_nodes: int = 500,
        show_labels: bool = True,
        label_field: str = "entity_name",
        physics: bool = True,
        height: str = "800px",
        tooltip_fields: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        """Visualize the complete knowledge graph.

        Args:
            graph_store: Graph document store with entity and relation data.
            output_path: Custom output path for the HTML file.
            layout: Layout algorithm ("hierarchical", "force", "circular").
            color_by: How to color nodes ("entity_type", "community", "degree").
            node_size: How to size nodes ("degree", "constant").
            filter_min_degree: Minimum degree threshold for nodes.
            filter_max_nodes: Maximum number of nodes to display.
            show_labels: Whether to show node labels.
            label_field: Field to use for labels.
            physics: Whether to enable physics simulation.
            height: Height of the visualization.
            tooltip_fields: Fields to include in tooltips.
            **kwargs: Additional parameters.

        Returns:
            Path to generated HTML file.
        """
        # Get NetworkX graph if available
        if hasattr(graph_store, "_graph"):
            nx_graph = graph_store._graph
        else:
            raise ValueError("Graph store does not expose NetworkX graph")

        # Filter nodes
        nodes_to_keep = self._filter_nodes(
            nx_graph, filter_min_degree, filter_max_nodes
        )

        # Create subgraph
        subgraph = nx_graph.subgraph(nodes_to_keep)

        # Create pyvis network
        net = Network(
            height=height,
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=False,
        )

        # Configure physics
        if physics:
            net.set_options(
                json.dumps({
                    "physics": {
                        "enabled": True,
                        "barnesHut": {
                            "gravitationalConstant": -8000,
                            "centralGravity": 0.3,
                            "springLength": 150,
                            "springConstant": 0.04,
                        },
                        "stabilization": {"iterations": 200},
                    }
                })
            )

        # Add nodes
        tooltip_fields = tooltip_fields or ["entity_type", "description"]
        for node_id in subgraph.nodes():
            node_data = subgraph.nodes[node_id]

            # Determine color
            if color_by == "entity_type":
                color = get_type_color(node_data.get("entity_type", "UNKNOWN"))
            elif color_by == "community":
                color = self._get_community_color(node_data)
            elif color_by == "degree":
                degree = subgraph.degree(node_id)
                max_degree = max(dict(subgraph.degree()).values())
                min_degree = min(dict(subgraph.degree()).values())
                color = get_gradient_color(degree, min_degree, max_degree)
            else:
                color = "#97c2fc"

            # Determine size
            if node_size == "degree":
                degree = subgraph.degree(node_id)
                size = min(50, max(10, degree * 2))
            else:
                size = 25

            # Create tooltip
            tooltip = self._create_tooltip(node_data, tooltip_fields)

            # Add node
            label = node_data.get(label_field, node_id) if show_labels else ""
            net.add_node(
                node_id,
                label=label,
                title=tooltip,
                size=size,
                color=color,
            )

        # Add edges
        for src, tgt in subgraph.edges():
            edge_data = subgraph.edges[src, tgt]
            weight = edge_data.get("weight", 1)
            width = max(1, min(10, weight * 2))

            net.add_edge(
                src,
                tgt,
                width=width,
                title=edge_data.get("description", ""),
            )

        # Set output path
        if output_path is None:
            output_path = str(self.output_dir / "knowledge_graph.html")

        # Save and return
        net.save_graph(output_path)
        return output_path

    def visualize_communities(
        self,
        communities: dict[str, Community],
        graph_store: GraphDocumentStore,
        output_path: Optional[str] = None,
        layout: str = "force",
        min_community_size: int = 3,
        show_community_labels: bool = True,
        show_entity_labels: bool = False,
        include_descriptions: bool = True,
        height: str = "800px",
        **kwargs,
    ) -> str:
        """Visualize community clustering structure.

        Args:
            communities: Dictionary of community ID to Community objects.
            graph_store: Graph document store.
            output_path: Custom output path for the HTML file.
            layout: Layout algorithm.
            min_community_size: Minimum community size to display.
            show_community_labels: Whether to show community-level labels.
            show_entity_labels: Whether to show entity-level labels.
            include_descriptions: Whether to include community descriptions.
            height: Height of the visualization.
            **kwargs: Additional parameters.

        Returns:
            Path to generated HTML file.
        """
        # Filter small communities
        filtered_communities = {
            k: v
            for k, v in communities.items()
            if len(v.nodes) >= min_community_size
        }

        # Generate colors
        colors = generate_community_colors(len(filtered_communities))

        # Create pyvis network
        net = Network(
            height=height,
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=False,
        )

        # Enable physics
        net.set_options(
            json.dumps({
                "physics": {
                    "enabled": True,
                    "barnesHut": {
                        "gravitationalConstant": -5000,
                        "centralGravity": 0.3,
                        "springLength": 200,
                    },
                }
            })
        )

        # Add nodes with community colors
        # Use enumerate for simple index-based color assignment
        for idx, (comm_id, community) in enumerate(filtered_communities.items()):
            color = colors[str(idx)]

            for entity_name in community.nodes:
                node_data = graph_store.get_node(entity_name)
                if node_data is None:
                    continue

                tooltip = (
                    f"<b>{entity_name}</b><br/>"
                    f"Type: {node_data.get('entity_type', 'UNKNOWN')}<br/>"
                    f"Community: {comm_id}"
                )

                label = entity_name if show_entity_labels else ""

                net.add_node(
                    entity_name,
                    label=label,
                    title=tooltip,
                    color=color,
                    size=20,
                )

        # Add edges
        if hasattr(graph_store, "_graph"):
            nx_graph = graph_store._graph
            for src, tgt in nx_graph.edges():
                if src in net.nodes and tgt in net.nodes:
                    edge_data = nx_graph.edges[src, tgt]
                    weight = edge_data.get("weight", 1)
                    width = max(1, min(10, weight * 2))
                    net.add_edge(src, tgt, width=width, hidden=False)

        # Set output path
        if output_path is None:
            output_path = str(self.output_dir / "communities.html")

        # Save and return
        net.save_graph(output_path)
        return output_path

    def visualize_query_path(
        self,
        path: list[str],
        graph_store: GraphDocumentStore,
        output_path: Optional[str] = None,
        show_context: int = 1,
        animate: bool = True,
        highlight_entities: Optional[list[str]] = None,
        layout: str = "hierarchical",
        include_relation_descriptions: bool = True,
        height: str = "600px",
        **kwargs,
    ) -> str:
        """Visualize a query path through the knowledge graph.

        Args:
            path: List of entity names forming the path.
            graph_store: Graph document store.
            output_path: Custom output path for the HTML file.
            show_context: Number of neighboring nodes to show.
            animate: Whether to animate the path.
            highlight_entities: Additional entities to highlight.
            layout: Layout algorithm.
            include_relation_descriptions: Whether to include relation descriptions.
            height: Height of the visualization.
            **kwargs: Additional parameters.

        Returns:
            Path to generated HTML file.
        """
        if not path:
            raise ValueError("Path cannot be empty")

        # Get NetworkX graph
        if hasattr(graph_store, "_graph"):
            nx_graph = graph_store._graph
        else:
            raise ValueError("Graph store does not expose NetworkX graph")

        # Extract subgraph with context
        nodes_to_include = set(path)

        if show_context > 0:
            for node in path:
                neighbors = list(nx_graph.neighbors(node))
                nodes_to_include.update(neighbors[:show_context])

        subgraph = nx_graph.subgraph(nodes_to_include)

        # Create pyvis network
        net = Network(
            height=height,
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=False,
        )

        # Configure hierarchical layout if requested
        if layout == "hierarchical":
            net.set_options(json.dumps({"layout": {"hierarchical": {"enabled": True}}}))

        # Highlight sets
        path_set = set(path)
        highlight_set = set(highlight_entities or [])

        # Add nodes
        for node_id in subgraph.nodes():
            node_data = subgraph.nodes[node_id]

            # Determine color based on role
            if node_id in path_set:
                color = "#ff6b6b"  # Red for path nodes
                size = 35
            elif node_id in highlight_set:
                color = "#ffd93d"  # Yellow for highlighted
                size = 30
            else:
                color = "#d3d3d3"  # Gray for context
                size = 20

            tooltip = f"<b>{node_id}</b><br/>Type: {node_data.get('entity_type', 'UNKNOWN')}"

            net.add_node(
                node_id,
                label=node_id,
                title=tooltip,
                color=color,
                size=size,
            )

        # Add edges
        for src, tgt in subgraph.edges():
            edge_data = subgraph.edges[src, tgt]
            weight = edge_data.get("weight", 1)

            # Highlight edges on the path
            edge_color = "#ff6b6b" if (src in path_set and tgt in path_set) else "#cccccc"
            width = 3 if (src in path_set and tgt in path_set) else 1

            title = edge_data.get("description", "") if include_relation_descriptions else ""

            net.add_edge(src, tgt, color=edge_color, width=width, title=title)

        # Set output path
        if output_path is None:
            output_path = str(self.output_dir / "query_path.html")

        # Save and return
        net.save_graph(output_path)
        return output_path

    def visualize_entity_stats(
        self,
        graph_store: GraphDocumentStore,
        output_path: Optional[str] = None,
        chart_types: Optional[list[str]] = None,
        show_top_n: int = 20,
        height: str = "600px",
        **kwargs,
    ) -> str:
        """Visualize entity extraction statistics.

        Args:
            graph_store: Graph document store.
            output_path: Custom output path for the HTML file.
            chart_types: Types of charts to generate ("distribution", "degree").
            show_top_n: Number of top entities to show in degree chart.
            height: Height of each chart.
            **kwargs: Additional parameters.

        Returns:
            Path to generated HTML file.
        """
        if chart_types is None:
            chart_types = ["distribution", "degree"]

        # Get NetworkX graph
        if hasattr(graph_store, "_graph"):
            nx_graph = graph_store._graph
        else:
            raise ValueError("Graph store does not expose NetworkX graph")

        # Create subplots with appropriate specs for different chart types
        n_charts = len(chart_types)

        # Determine subplot specs based on chart types
        specs = []
        for chart_type in chart_types:
            if chart_type == "distribution":
                specs.append([{"type": "pie"}])
            else:
                specs.append([{"type": "xy"}])

        fig = make_subplots(
            rows=n_charts,
            cols=1,
            subplot_titles=[self._get_chart_title(ct) for ct in chart_types],
            vertical_spacing=0.15,
            specs=specs,
        )

        # Generate charts
        row = 1
        for chart_type in chart_types:
            if chart_type == "distribution":
                chart_fig = self._create_type_distribution_chart(nx_graph)
            elif chart_type == "degree":
                chart_fig = self._create_degree_distribution_chart(
                    nx_graph, show_top_n
                )
            else:
                continue

            # Add to subplot
            for trace in chart_fig.data:
                fig.add_trace(trace, row=row, col=1)
            row += 1

        # Update layout
        fig.update_layout(
            height=int(height.replace("px", "")) * n_charts + 100 * n_charts,
            showlegend=True,
            title_text="HiRAG Entity Statistics",
        )

        # Set output path
        if output_path is None:
            output_path = str(self.output_dir / "entity_stats.html")

        # Save and return
        fig.write_html(output_path)
        return output_path

    def visualize_all(
        self,
        graph_store: GraphDocumentStore,
        communities: Optional[dict[str, Community]] = None,
        **kwargs,
    ) -> dict[str, str]:
        """Generate all visualizations at once.

        Args:
            graph_store: Graph document store.
            communities: Dictionary of communities (optional, will detect if not provided).
            **kwargs: Additional parameters for specific visualizations.

        Returns:
            Dictionary mapping visualization names to file paths.
        """
        results = {}

        # Knowledge graph
        results["knowledge_graph"] = self.visualize_knowledge_graph(
            graph_store, **kwargs
        )

        # Communities
        if communities is None:
            communities = graph_store.clustering()
        results["communities"] = self.visualize_communities(
            communities, graph_store, **kwargs
        )

        # Statistics
        results["statistics"] = self.visualize_entity_stats(graph_store, **kwargs)

        return results

    # ===== Private Helper Methods =====

    def _filter_nodes(
        self, nx_graph: nx.Graph, min_degree: int, max_nodes: int
    ) -> set:
        """Filter nodes based on degree and count.

        Args:
            nx_graph: NetworkX graph.
            min_degree: Minimum degree threshold.
            max_nodes: Maximum number of nodes to return.

        Returns:
            Set of node IDs to keep.
        """
        # Filter by degree
        nodes_by_degree = {
            node: degree for node, degree in nx_graph.degree() if degree >= min_degree
        }

        # Limit to top N nodes by degree
        if len(nodes_by_degree) > max_nodes:
            sorted_nodes = sorted(
                nodes_by_degree.items(), key=lambda x: x[1], reverse=True
            )
            nodes_by_degree = dict(sorted_nodes[:max_nodes])

        return set(nodes_by_degree.keys())

    def _create_tooltip(self, node_data: dict, fields: list[str]) -> str:
        """Create HTML tooltip for a node.

        Args:
            node_data: Node data dictionary.
            fields: Fields to include in tooltip.

        Returns:
            HTML string for tooltip.
        """
        lines = [f"<b>{node_data.get('entity_name', 'Unknown')}</b>"]
        for field in fields:
            value = node_data.get(field, "")
            if value:
                lines.append(f"{field}: {value}")
        return "<br/>".join(lines)

    def _get_community_color(self, node_data: dict) -> str:
        """Get color for a node based on its community assignment.

        Args:
            node_data: Node data dictionary.

        Returns:
            Hex color string.
        """
        clusters = node_data.get("clusters", [])
        if clusters and len(clusters) > 0:
            # Handle both dict and string cluster assignments
            first_cluster = clusters[0]
            if isinstance(first_cluster, dict):
                cluster_id = first_cluster.get("cluster", "0")
            else:
                cluster_id = str(first_cluster)

            # Generate consistent color from cluster ID
            colors = generate_community_colors(100)
            return colors.get(str(cluster_id), "#97c2fc")
        return "#97c2fc"

    def _get_chart_title(self, chart_type: str) -> str:
        """Get title for a chart type.

        Args:
            chart_type: Type of chart.

        Returns:
            Chart title.
        """
        titles = {
            "distribution": "Entity Type Distribution",
            "degree": "Top Entities by Connection Count",
            "overview": "Graph Overview",
        }
        return titles.get(chart_type, chart_type)

    def _create_type_distribution_chart(self, nx_graph: nx.Graph) -> go.Figure:
        """Create pie chart of entity type distribution.

        Args:
            nx_graph: NetworkX graph.

        Returns:
            Plotly figure.
        """
        type_counts = defaultdict(int)
        for node_id in nx_graph.nodes():
            node_data = nx_graph.nodes[node_id]
            entity_type = node_data.get("entity_type", "UNKNOWN")
            type_counts[entity_type] += 1

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(type_counts.keys()),
                    values=list(type_counts.values()),
                    hole=0.3,
                )
            ]
        )
        fig.update_layout(title="Entity Type Distribution")
        return fig

    def _create_degree_distribution_chart(
        self, nx_graph: nx.Graph, top_n: int
    ) -> go.Figure:
        """Create bar chart of top entities by degree.

        Args:
            nx_graph: NetworkX graph.
            top_n: Number of top entities to show.

        Returns:
            Plotly figure.
        """
        degrees = {node: nx_graph.degree(node) for node in nx_graph.nodes()}
        top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=[name for name, _ in top_entities],
                    y=[deg for _, deg in top_entities],
                )
            ]
        )
        fig.update_layout(
            title=f"Top {top_n} Entities by Connection Count",
            xaxis_title="Entity",
            yaxis_title="Degree",
        )
        return fig

    def _create_overview_dashboard(
        self, nx_graph: nx.Graph, top_n: int
    ) -> go.Figure:
        """Create overview dashboard with multiple metrics.

        Args:
            nx_graph: NetworkX graph.
            top_n: Number of top entities for metrics.

        Returns:
            Plotly figure with subplots.
        """
        # Calculate metrics
        n_nodes = nx_graph.number_of_nodes()
        n_edges = nx_graph.number_of_edges()
        avg_degree = sum(dict(nx_graph.degree()).values()) / n_nodes if n_nodes > 0 else 0

        # Create indicator charts
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["Total Entities", "Total Relations", "Avg Connections"],
        )

        fig.add_trace(
            go.Indicator(mode="number", value=n_nodes), row=1, col=1
        )
        fig.add_trace(
            go.Indicator(mode="number", value=n_edges), row=1, col=2
        )
        fig.add_trace(
            go.Indicator(mode="number", value=f"{avg_degree:.1f}"), row=1, col=3
        )

        fig.update_layout(title="Graph Overview")
        return fig
