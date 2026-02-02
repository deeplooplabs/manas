"""ManasRAG Streamlit WebUI.

A user-friendly interface for the ManasRAG hierarchical retrieval system.
Supports document indexing, querying with multiple retrieval modes,
and knowledge graph visualization.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils.auth import Secret
from haystack.dataclasses import Document

from manasrag import (
    ManasRAG,
    QueryParam,
    RetrievalMode,
    DocumentLoader,
)
from manasrag.stores import EntityVectorStore, ChunkVectorStore

# ===== Page Configuration =====
st.set_page_config(
    page_title="ManasRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== CSS Styling =====
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .context-text {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        max-height: 300px;
        overflow-y: auto;
        font-size: 0.9rem;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


# ===== Session State Initialization =====
def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "manas" not in st.session_state:
        st.session_state.manas = None
    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_project" not in st.session_state:
        st.session_state.current_project = "default"


init_session_state()


# ===== Configuration Management =====
@st.cache_resource
def load_configuration() -> dict[str, Any]:
    """Load configuration from environment and sidebar settings.

    Returns:
        Dictionary with configuration values.
    """
    load_dotenv()

    return {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "working_dir": st.session_state.get("working_dir", "./manas_data_ui"),
        "top_k": st.session_state.get("top_k", 20),
        "top_m": st.session_state.get("top_m", 10),
        "chunk_size": st.session_state.get("chunk_size", 1200),
        "chunk_overlap": st.session_state.get("chunk_overlap", 100),
    }


def initialize_manas_rag(config: dict[str, Any]) -> ManasRAG | None:
    """Initialize ManasRAG instance with current configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        ManasRAG instance or None if configuration is invalid.
    """
    if not config["api_key"]:
        return None

    if st.session_state.manas is None:
        try:
            # Initialize generator
            if config["base_url"]:
                generator = OpenAIChatGenerator(
                    api_key=Secret.from_token(config["api_key"]),
                    model=config["model"],
                    api_base_url=config["base_url"],
                    timeout=120.0,
                )
            else:
                generator = OpenAIChatGenerator(
                    api_key=Secret.from_token(config["api_key"]),
                    model=config["model"],
                    timeout=120.0,
                )

            # Set up stores
            chunk_store = ChunkVectorStore(working_dir=config["working_dir"])
            entity_store = EntityVectorStore(working_dir=config["working_dir"])

            # Initialize ManasRAG
            manas = ManasRAG(
                working_dir=config["working_dir"],
                generator=generator,
                entity_store=entity_store,
                chunk_store=chunk_store,
                top_k=config["top_k"],
                top_m=config["top_m"],
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                log_level="INFO",
            )

            st.session_state.manas = manas
            return manas

        except Exception as e:
            st.error(f"Failed to initialize ManasRAG: {e}")
            return None

    return st.session_state.manas


# ===== Sidebar =====
def render_sidebar() -> dict[str, Any]:
    """Render the sidebar with configuration options.

    Returns:
        Configuration dictionary.
    """
    with st.sidebar:
        st.markdown("# ⚙️ Configuration")

        # API Configuration
        st.markdown("## API Settings")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Your OpenAI API key. Can also be set via OPENAI_API_KEY environment variable.",
        )
        base_url = st.text_input(
            "Base URL (Optional)",
            value=os.getenv("OPENAI_BASE_URL", ""),
            help="Custom API base URL. Leave empty for default OpenAI endpoint.",
        )
        model = st.text_input(
            "Model",
            value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            help="Model name to use for generation and embedding.",
        )

        # Working Directory
        st.markdown("## Storage Settings")
        working_dir = st.text_input(
            "Working Directory",
            value="./manas_data_ui",
            help="Directory for storing indexed data and cache.",
        )

        # Retrieval Parameters
        st.markdown("## Retrieval Parameters")
        top_k = st.slider(
            "Top K Entities",
            min_value=5,
            max_value=50,
            value=20,
            help="Number of entities to retrieve for local context.",
        )
        top_m = st.slider(
            "Top M per Community",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of key entities per community for bridge paths.",
        )

        # Chunking Parameters
        st.markdown("## Chunking Parameters")
        chunk_size = st.slider(
            "Chunk Size",
            min_value=500,
            max_value=2000,
            value=1200,
            step=100,
            help="Token size for document chunking.",
        )
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=300,
            value=100,
            step=50,
            help="Token overlap between chunks.",
        )

        # Store in session state
        st.session_state.working_dir = working_dir
        st.session_state.top_k = top_k
        st.session_state.top_m = top_m
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap

        # Project Management
        st.markdown("---")
        st.markdown("## 📁 Project Management")
        project_id = st.text_input(
            "Current Project",
            value=st.session_state.current_project,
            help="Project ID for data isolation.",
        )
        if st.button("Switch Project"):
            st.session_state.current_project = project_id
            st.session_state.manas = None
            st.session_state.indexed_documents = {}
            st.session_state.chat_history = []
            st.rerun()

        # About
        st.markdown("---")
        st.markdown("## About")
        st.markdown("""
        **ManasRAG** v0.1.1

        Hierarchical Retrieval-Augmented Generation
        powered by Haystack.

        [GitHub](https://github.com/deeplooplabs/manasrag)
        """)

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "working_dir": working_dir,
        "top_k": top_k,
        "top_m": top_m,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


# ===== Document Indexing Tab =====
def render_indexing_tab(manas: ManasRAG | None) -> None:
    """Render the document indexing tab.

    Args:
        manas: ManasRAG instance or None.
    """
    st.markdown('<div class="sub-header">📄 Document Indexing</div>', unsafe_allow_html=True)

    if manas is None:
        st.warning("Please configure your API key in the sidebar to begin indexing.")
        return

    # Document Input Methods
    input_method = st.radio(
        "Select input method:",
        ["Upload Files", "Paste Text", "Load from Directory"],
        horizontal=True,
    )

    documents: list[Document] = []

    if input_method == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "docx", "html", "csv"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, MD, DOCX, HTML, CSV",
        )

        if uploaded_files:
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{file.name.split('.')[-1]}",
                ) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name

                try:
                    loader = DocumentLoader(verbose=True)
                    docs = loader.load([tmp_path])
                    documents.extend(docs)
                    st.success(f"Loaded {file.name}: {len(docs)} document(s)")
                except Exception as e:
                    st.error(f"Error loading {file.name}: {e}")
                finally:
                    os.unlink(tmp_path)

    elif input_method == "Paste Text":
        text_content = st.text_area(
            "Paste document text",
            height=300,
            help="Enter the text content you want to index.",
        )

        doc_id = st.text_input(
            "Document ID (Optional)",
            placeholder="Leave empty to auto-generate",
            help="A unique identifier for this document.",
        )

        if text_content and st.button("Index Text"):
            documents = [Document(content=text_content, id=doc_id or None)]

    elif input_method == "Load from Directory":
        dir_path = st.text_input(
            "Directory Path",
            placeholder="./docs",
            help="Path to directory containing documents.",
        )

        file_pattern = st.text_input(
            "File Pattern",
            value="**/*",
            help="Glob pattern for matching files (e.g., **/*.pdf).",
        )

        if st.button("Load from Directory"):
            if dir_path:
                try:
                    full_pattern = os.path.join(dir_path, file_pattern)
                    loader = DocumentLoader(verbose=True)
                    documents = loader.load([full_pattern])
                    st.success(f"Loaded {len(documents)} document(s)")
                except Exception as e:
                    st.error(f"Error loading from directory: {e}")
            else:
                st.warning("Please enter a directory path.")

    # Indexing Options
    if documents:
        st.markdown("---")
        st.markdown("### Indexing Options")

        col1, col2 = st.columns(2)
        with col1:
            incremental = st.checkbox(
                "Incremental Indexing",
                value=False,
                help="Only index new documents.",
            )
        with col2:
            force_reindex = st.checkbox(
                "Force Re-index",
                value=False,
                help="Re-index all documents.",
            )

        if st.button("🚀 Start Indexing", type="primary", use_container_width=True):
            with st.spinner("Indexing documents... This may take a while."):
                try:
                    project_id = st.session_state.current_project
                    result = manas.index(
                        documents,
                        project_id=project_id,
                        incremental=incremental,
                        force_reindex=force_reindex,
                    )

                    # Store indexed documents info
                    for doc in documents:
                        doc_id = doc.id or "unknown"
                        st.session_state.indexed_documents[doc_id] = {
                            "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        }

                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("#### Indexing Complete!")
                    st.json(result)
                    st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    # Display Indexed Documents
    if st.session_state.indexed_documents:
        st.markdown("---")
        st.markdown("### 📚 Indexed Documents")

        for doc_id, info in st.session_state.indexed_documents.items():
            with st.expander(f"Document: {doc_id[:16]}..."):
                st.text(info["content"])


# ===== Query Tab =====
def render_query_tab(manas: ManasRAG | None) -> None:
    """Render the query tab.

    Args:
        manas: ManasRAG instance or None.
    """
    st.markdown('<div class="sub-header">🔍 Query</div>', unsafe_allow_html=True)

    if manas is None:
        st.warning("Please configure your API key in the sidebar to begin querying.")
        return

    if not st.session_state.indexed_documents:
        st.info("No documents indexed yet. Go to the Indexing tab to add documents.")
        return

    # Retrieval Mode Selection
    st.markdown("### Retrieval Mode")

    mode_descriptions = {
        "naive": "Basic RAG with document chunks only",
        "local": "Local knowledge: entities + relations + chunks",
        "global": "Global knowledge: community reports + chunks",
        "bridge": "Bridge knowledge: cross-community reasoning paths",
        "nobridge": "Hierarchical without bridge paths (faster)",
        "hi": "Full hierarchical: all of the above",
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        mode = st.selectbox(
            "Select Mode",
            options=list(mode_descriptions.keys()),
            index=5,
            format_func=lambda x: x.upper(),
        )
    with col2:
        st.markdown(f"**Description:** {mode_descriptions[mode]}")

    # Advanced Parameters
    with st.expander("Advanced Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Top K", 5, 50, 20)
            top_m = st.slider("Top M", 5, 20, 10)
        with col2:
            level = st.slider("Community Level", 0, 5, 2)
            response_type = st.selectbox(
                "Response Type",
                ["Multiple Paragraphs", "Single Paragraph", "Bullet Points"],
            )

        only_context = st.checkbox("Return Context Only (No Generation)", value=False)

    # Query Input
    st.markdown("---")
    st.markdown("### Ask a Question")

    query = st.text_area(
        "Enter your question",
        placeholder="What are the main topics covered in the documents?",
        height=100,
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("🚀 Query", type="primary", use_container_width=True)
    with col2:
        clear_history = st.button("🗑️ Clear History")

    if clear_history:
        st.session_state.chat_history = []
        st.rerun()

    # Display Chat History
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Conversation History")

        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)

    # Process Query
    if submit and query:
        if not manas:
            st.error("ManasRAG not initialized. Check your API configuration.")
            return

        with st.spinner("Processing query..."):
            try:
                # Build query parameters
                param = QueryParam(
                    mode=mode,
                    top_k=top_k,
                    top_m=top_m,
                    level=level,
                    response_type=response_type,
                    only_need_context=only_context,
                )

                project_id = st.session_state.current_project
                result = manas.query(query, mode=mode, param=param, project_id=project_id)

                # Display result
                st.markdown("---")
                st.markdown('<div class="sub-header">Answer</div>', unsafe_allow_html=True)
                st.markdown(result["answer"])

                # Display context
                if "context" in result and result["context"]:
                    with st.expander("View Retrieved Context"):
                        st.markdown('<div class="context-text">', unsafe_allow_html=True)
                        st.text(result["context"])
                        st.markdown('</div>', unsafe_allow_html=True)

                # Add to history
                st.session_state.chat_history.append((query, result["answer"]))

            except Exception as e:
                st.error(f"Query failed: {e}")


# ===== Visualization Tab =====
def render_visualization_tab(manas: ManasRAG | None) -> None:
    """Render the visualization tab.

    Args:
        manas: ManasRAG instance or None.
    """
    st.markdown('<div class="sub-header">📊 Knowledge Graph Visualization</div>', unsafe_allow_html=True)

    if manas is None:
        st.warning("Please configure your API key in the sidebar to enable visualization.")
        return

    if not st.session_state.indexed_documents:
        st.info("No documents indexed yet. Go to the Indexing tab to add documents.")
        return

    # Visualization Type
    viz_type = st.radio(
        "Visualization Type",
        ["Knowledge Graph", "Communities", "Entity Statistics", "All"],
        horizontal=True,
    )

    # Visualization Options
    with st.expander("Visualization Options"):
        col1, col2 = st.columns(2)
        with col1:
            layout = st.selectbox("Layout", ["force", "hierarchical", "circular"])
            color_by = st.selectbox("Color By", ["entity_type", "community", "degree"])
        with col2:
            show_labels = st.checkbox("Show Labels", value=True)
            physics = st.checkbox("Enable Physics", value=True)
            filter_min_degree = st.slider("Min Degree Filter", 0, 5, 0)

    if st.button("Generate Visualization", type="primary"):
        with st.spinner("Generating visualization..."):
            try:
                kind_map = {
                    "Knowledge Graph": "graph",
                    "Communities": "communities",
                    "Entity Statistics": "stats",
                    "All": "all",
                }

                result = manas.visualize(
                    kind=kind_map[viz_type],
                    project_id=st.session_state.current_project,
                    layout=layout,
                    color_by=color_by,
                    show_labels=show_labels,
                    physics=physics,
                    filter_min_degree=filter_min_degree,
                )

                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("#### Visualization Generated!")
                st.markdown('</div>', unsafe_allow_html=True)

                # Display results
                for name, path in result.items():
                    st.markdown(f"**{name.title()}:** `{path}`")

                    # Try to display HTML
                    if path and Path(path).exists():
                        with open(path, "r", encoding="utf-8") as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=600, scrolling=True)

            except Exception as e:
                st.error(f"Visualization failed: {e}")


# ===== Statistics Tab =====
def render_statistics_tab(manas: ManasRAG | None) -> None:
    """Render the statistics tab.

    Args:
        manas: ManasRAG instance or None.
    """
    st.markdown('<div class="sub-header">📈 Statistics</div>', unsafe_allow_html=True)

    if manas is None:
        st.warning("Please configure your API key to view statistics.")
        return

    project_id = st.session_state.current_project

    # Get graph store
    try:
        graph_store = manas.get_graph_store(project_id=project_id)

        st.markdown("### Graph Statistics")

        # Node statistics
        node_count = graph_store.node_count()
        edge_count = graph_store.edge_count()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes", node_count)
        with col2:
            st.metric("Total Edges", edge_count)
        with col3:
            st.metric("Avg Degree", round(edge_count * 2 / node_count, 2) if node_count > 0 else 0)

        # Community statistics
        st.markdown("### Community Statistics")

        communities = manas.communities
        if communities:
            num_communities = len(communities)
            st.metric("Total Communities", num_communities)

            # Community size distribution
            community_sizes = [len(c.entities) for c in communities.values()]

            if community_sizes:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max Community Size", max(community_sizes))
                with col2:
                    st.metric("Min Community Size", min(community_sizes))
                with col3:
                    st.metric("Avg Community Size", round(sum(community_sizes) / len(community_sizes), 1))

        # Document statistics
        st.markdown("### Document Statistics")

        indexed_docs = st.session_state.indexed_documents
        st.metric("Indexed Documents", len(indexed_docs))

        # Project info
        st.markdown("---")
        st.markdown("### Project Information")

        st.json({
            "project_id": project_id,
            "working_dir": manas.working_dir,
            "graph_backend": manas.graph_backend,
            "top_k": manas.top_k,
            "top_m": manas.top_m,
        })

    except Exception as e:
        st.error(f"Failed to load statistics: {e}")


# ===== Main Application =====
def main() -> None:
    """Main application entry point."""
    # Render header
    st.markdown('<div class="main-header">🧠 ManasRAG</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 1.1rem; color: #666;'>"
        "Hierarchical Retrieval-Augmented Generation with Knowledge Graphs"
        "</p>",
        unsafe_allow_html=True,
    )

    # Render sidebar
    config = render_sidebar()

    # Initialize ManasRAG
    manas = initialize_manas_rag(config)

    # Check API configuration
    if not config["api_key"]:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ### ⚠️ API Key Required

        Please configure your OpenAI API key in the sidebar to get started.

        You can set the `OPENAI_API_KEY` environment variable or enter it directly in the sidebar.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Indexing",
        "🔍 Query",
        "📊 Visualization",
        "📈 Statistics",
    ])

    with tab1:
        render_indexing_tab(manas)

    with tab2:
        render_query_tab(manas)

    with tab3:
        render_visualization_tab(manas)

    with tab4:
        render_statistics_tab(manas)


if __name__ == "__main__":
    main()
