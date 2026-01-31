"""Visualization examples for ManasRAG.

This example demonstrates:
1. Visualizing the complete knowledge graph
2. Visualizing community structure
3. Visualizing query paths
4. Visualizing entity statistics
5. Generating all visualizations at once
"""

import os
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

from haystack.components.generators.chat import OpenAIChatGenerator

from manasrag import ManasRAG
from manasrag.components import GraphVisualizer
from manasrag.stores import EntityVectorStore, ChunkVectorStore


def visualize_knowledge_graph_example():
    """Example: Visualize the complete knowledge graph."""
    print("\n" + "=" * 60)
    print("1. Knowledge Graph Visualization")
    print("=" * 60)

    # Load or create ManasRAG instance
    working_dir = "./manas_data"
    if not Path(working_dir).exists():
        print(f"No data found at {working_dir}")
        print("Please run basic_usage.py first to create some data.")
        return None

    # Initialize ManasRAG
    manas = ManasRAG(working_dir=working_dir)

    # Create visualizer
    visualizer = GraphVisualizer(
        output_dir="./manas_visualizations",
        default_layout="hierarchical",
    )

    # Generate knowledge graph visualization
    html_path = visualizer.visualize_knowledge_graph(
        graph_store=manas.graph_store,
        layout="force",
        color_by="entity_type",
        show_labels=True,
        physics=True,
        height="900px",
    )

    print(f"[OK] Knowledge graph saved to: {html_path}")
    print("  - Open the HTML file in your browser to explore the graph")
    print("  - You can drag nodes, zoom in/out, and hover for details")

    return html_path


def visualize_communities_example():
    """Example: Visualize community structure."""
    print("\n" + "=" * 60)
    print("2. Community Visualization")
    print("=" * 60)

    # Load ManasRAG instance
    working_dir = "./manas_data"
    if not Path(working_dir).exists():
        print(f"No data found at {working_dir}")
        return None

    manas = ManasRAG(working_dir=working_dir)
    visualizer = GraphVisualizer()

    # Detect communities if not already done
    if not manas.communities:
        print("Detecting communities...")
        manas.communities = manas.graph_store.clustering()

    # Visualize communities
    html_path = visualizer.visualize_communities(
        communities=manas.communities,
        graph_store=manas.graph_store,
        layout="force",
        show_community_labels=True,
        show_entity_labels=False,
        min_community_size=3,
        include_descriptions=True,
    )

    print(f"[OK] Community visualization saved to: {html_path}")
    print(f"  - Found {len(manas.communities)} communities")

    return html_path


def visualize_query_path_example():
    """Example: Visualize a query path."""
    print("\n" + "=" * 60)
    print("3. Query Path Visualization")
    print("=" * 60)

    # Load ManasRAG instance
    working_dir = "./manas_data"
    if not Path(working_dir).exists():
        print(f"No data found at {working_dir}")
        return None

    manas = ManasRAG(working_dir=working_dir)
    visualizer = GraphVisualizer()

    # Find a path between two entities
    # You can customize these entity names based on your data
    source = "ARTIFICIAL INTELLIGENCE"
    target = "KNOWLEDGE GRAPHS"

    print(f"Finding path from '{source}' to '{target}'...")

    path = manas.graph_store.shortest_path(source, target)

    if path and len(path) > 1:
        html_path = visualizer.visualize_query_path(
            path=path,
            graph_store=manas.graph_store,
            show_context=1,  # Show 1 level of neighboring nodes
            animate=True,
            include_relation_descriptions=True,
        )

        print(f"[OK] Path visualization saved to: {html_path}")
        print(f"  - Path length: {len(path)} entities")
        print(f"  - Path: {' → '.join(path)}")
    else:
        print(f"[ERROR] No path found between '{source}' and '{target}'")
        print("  Try different entity names or run basic_usage.py first")
        return None

    return html_path


def visualize_statistics_example():
    """Example: Visualize entity statistics."""
    print("\n" + "=" * 60)
    print("4. Entity Statistics Visualization")
    print("=" * 60)

    # Load ManasRAG instance
    working_dir = "./manas_data"
    if not Path(working_dir).exists():
        print(f"No data found at {working_dir}")
        return None

    manas = ManasRAG(working_dir=working_dir)
    visualizer = GraphVisualizer()

    # Generate statistics dashboard
    html_path = visualizer.visualize_entity_stats(
        graph_store=manas.graph_store,
        chart_types=["distribution", "degree"],
        show_top_n=20,
    )

    print(f"[OK] Statistics dashboard saved to: {html_path}")
    print("  - Contains entity type distribution, top entities, and overview")

    return html_path


def visualize_all_example():
    """Example: Generate all visualizations at once."""
    print("\n" + "=" * 60)
    print("5. Generate All Visualizations")
    print("=" * 60)

    # Load ManasRAG instance
    working_dir = "./manas_data"
    if not Path(working_dir).exists():
        print(f"No data found at {working_dir}")
        return {}

    manas = ManasRAG(working_dir=working_dir)
    visualizer = GraphVisualizer(output_dir="./manas_visualizations")

    # Detect communities if needed
    if not manas.communities:
        print("Detecting communities...")
        manas.communities = manas.graph_store.clustering()

    # Generate all visualizations
    results = visualizer.visualize_all(
        graph_store=manas.graph_store,
        communities=manas.communities,
    )

    print("\nGenerated visualizations:")
    for name, path in results.items():
        print(f"  [OK] {name}: {path}")

    return results


def visualize_with_customization_example():
    """Example: Advanced visualization with custom options."""
    print("\n" + "=" * 60)
    print("6. Customized Visualization")
    print("=" * 60)

    # Load ManasRAG instance
    working_dir = "./manas_data"
    if not Path(working_dir).exists():
        print(f"No data found at {working_dir}")
        return None

    manas = ManasRAG(working_dir=working_dir)
    visualizer = GraphVisualizer()

    # Custom knowledge graph with advanced options
    html_path = visualizer.visualize_knowledge_graph(
        graph_store=manas.graph_store,
        layout="force",
        color_by="community",  # Color by community instead of type
        node_size="degree",  # Size by degree
        filter_min_degree=1,  # Only show entities with 1+ connections
        filter_max_nodes=100,  # Limit to 100 most connected entities
        show_labels=True,
        tooltip_fields=["entity_name", "entity_type", "description"],
        physics=True,
        height="1000px",
    )

    print(f"[OK] Custom visualization saved to: {html_path}")
    print("  - Colored by community")
    print("  - Sized by connection count")
    print("  - Filtered to top 100 connected entities")

    return html_path


def main():
    """Run all visualization examples."""
    # Load environment variables
    load_dotenv()

    # Debug: Print environment variables
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    print("=" * 60)
    print("HiRAG-Haystack Visualization Examples")
    print("=" * 60)
    print("\nLoaded from .env:")
    if api_key:
        print(f"  OPENAI_API_KEY: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else api_key}")
    else:
        print(f"  OPENAI_API_KEY: (not set)")
    print(f"  OPENAI_BASE_URL: {base_url}")
    print(f"  OPENAI_MODEL: {model}")
    print()
    print("These examples require data from basic_usage.py")
    print("If no data is found, please run: python examples/basic_usage.py\n")

    # Check for existing data
    if not Path("./manas_data").exists():
        print("[!] No HiRAG data found!")
        print("Please run basic_usage.py first to create sample data.")
        print("\nYou can also use your own data by setting working_dir parameter.")
        return

    try:
        # Run examples
        results = {}

        kg_path = visualize_knowledge_graph_example()
        if kg_path:
            results["knowledge_graph"] = kg_path

        comm_path = visualize_communities_example()
        if comm_path:
            results["communities"] = comm_path

        path_path = visualize_query_path_example()
        if path_path:
            results["query_path"] = path_path

        stats_path = visualize_statistics_example()
        if stats_path:
            results["statistics"] = stats_path

        all_results = visualize_all_example()
        results.update(all_results)

        custom_path = visualize_with_customization_example()
        if custom_path:
            results["custom"] = custom_path

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Generated {len(results)} visualization(s)")

        # Optionally open in browser
        print("\nWould you like to open the knowledge graph in your browser?")
        response = input("Enter 'y' to open, or any other key to skip: ").strip().lower()

        if response == 'y' and "knowledge_graph" in results:
            # Convert to absolute path
            kg_path = str(Path(results["knowledge_graph"]).resolve())
            print(f"\nOpening {kg_path}...")
            webbrowser.open(f"file:///{kg_path}")

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
