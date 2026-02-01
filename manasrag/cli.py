"""CLI entry point for ManasRAG.

This module provides a command-line interface for indexing documents,
querying the ManasRAG knowledge graph, and visualizing results.
"""

import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any

import click
import yaml
from dotenv import load_dotenv
from haystack.utils.auth import Secret

from manasrag import ManasRAG, QueryParam, RetrievalMode, __version__
from manasrag._logging import setup_logging
from manasrag.components import GraphVisualizer
from manasrag.document_loader import DocumentLoader
from manasrag.stores import EntityVectorStore, ChunkVectorStore


# ===== CONFIG FILE HANDLING =====

def _find_config_file() -> Path | None:
    """Find config file in default locations.

    Search order:
    1. ./manas.yaml
    2. ~/.manas.yaml

    Returns:
        Path to config file if found, None otherwise.
    """
    candidates = [
        Path("./manas.yaml"),
        Path.home() / ".manas.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_config(config_path: Path | None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file, or None to auto-detect.

    Returns:
        Configuration dictionary. Empty dict if no config found.
    """
    if config_path is None:
        config_path = _find_config_file()

    if config_path is None or not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except Exception as e:
        click.echo(f"Warning: Failed to load config from {config_path}: {e}", err=True)
        return {}


def _resolve_value(
    cli_value: Any,
    config_value: Any,
    env_var: str | None = None,
    default: Any = None,
) -> Any:
    """Resolve a configuration value with precedence.

    Precedence order:
    1. CLI flag (if explicitly provided, i.e., not None)
    2. Config file value
    3. Environment variable
    4. Default value

    Args:
        cli_value: Value from CLI flag.
        config_value: Value from config file.
        env_var: Environment variable name to check.
        default: Default value if all else fails.

    Returns:
        Resolved value.
    """
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value
    return default


# ===== MANAS INSTANCE BUILDER =====

class ManasRAGConfigError(Exception):
    """Raised when ManasRAG configuration is invalid."""
    pass


def _build_manasrag(
    working_dir: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    graph_backend: str,
    chunk_size: int,
    chunk_overlap: int,
    timeout: int | None,
    top_k: int = 20,
    top_m: int = 10,
    verbose: bool = False,
    project_id: str | None = None,
) -> ManasRAG:
    """Build a ManasRAG instance with the given configuration.

    Args:
        working_dir: Working directory for cache and data.
        model: LLM model name.
        api_key: OpenAI API key.
        base_url: API base URL.
        graph_backend: Graph backend ("networkx" or "neo4j").
        chunk_size: Token chunk size.
        chunk_overlap: Chunk overlap.
        timeout: LLM timeout in seconds.
        top_k: Number of entities to retrieve.
        top_m: Key entities per community.
        verbose: Print progress information.

    Returns:
        Configured ManasRAG instance.

    Raises:
        ManasRAGConfigError: If API key is missing or OpenAI integration is not installed.
    """
    # Resolve API key from environment if not provided
    api_key = api_key or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ManasRAGConfigError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or use --api-key flag."
        )

    # Create generator
    try:
        from haystack.components.generators.chat import OpenAIChatGenerator
    except ImportError:
        raise ManasRAGConfigError(
            "OpenAI integration not installed. Run: pip install manas[openai]"
        )

    generator_kwargs: dict[str, Any] = {}
    if api_key:
        generator_kwargs["api_key"] = Secret.from_token(api_key)
    if model:
        generator_kwargs["model"] = model
    if base_url:
        generator_kwargs["api_base_url"] = base_url
    if timeout:
        generator_kwargs["timeout"] = timeout

    generator = OpenAIChatGenerator(**generator_kwargs)

    # Create vector stores
    entity_store = EntityVectorStore(
        working_dir=working_dir,
    )
    chunk_store = ChunkVectorStore(
        working_dir=working_dir,
    )

    # Create ManasRAG instance
    manas = ManasRAG(
        working_dir=working_dir,
        graph_backend=graph_backend,
        generator=generator,
        entity_store=entity_store,
        chunk_store=chunk_store,
        top_k=top_k,
        top_m=top_m,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return manas


# ===== CLI COMMANDS =====

@click.group()
@click.option(
    "-d", "--working-dir",
    type=click.Path(),
    default=None,
    help="Working directory for cache and data (default: ./manas_data).",
)
@click.option(
    "-c", "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Config file path (default: auto-detect manas.yaml).",
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Enable verbose output (INFO level).",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="WARNING",
    help="Logging level (default: WARNING).",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="LLM timeout in seconds (default: 600).",
)
@click.version_option(version=__version__, prog_name="manas")
@click.pass_context
def cli(ctx: click.Context, working_dir: str | None, config: Path | None, verbose: bool, log_level: str, timeout: int | None) -> None:
    """ManasRAG: Hierarchical Retrieval-Augmented Generation CLI.

    Index documents and query knowledge graphs using the ManasRAG system.
    """
    # Load .env file
    load_dotenv()

    # Configure logging (default WARNING, INFO if verbose)
    log_level_value = "INFO" if verbose else log_level
    setup_logging(level=log_level_value)

    # Load config
    config_data = _load_config(config)

    # Store in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["config"] = config_data
    ctx.obj["working_dir"] = _resolve_value(
        working_dir,
        config_data.get("working_dir"),
        default="./manas_data",
    )
    ctx.obj["verbose"] = verbose
    ctx.obj["log_level"] = log_level
    ctx.obj["timeout"] = _resolve_value(
        timeout,
        config_data.get("timeout"),
        env_var="MANAS_TIMEOUT",
        default=600,
    )


@cli.command(name="add-documents")
@click.argument("sources", nargs=-1, required=True)
@click.option(
    "--incremental/--no-incremental",
    default=None,
    help="Skip already-indexed documents (default: on).",
)
@click.option(
    "--force-reindex",
    is_flag=True,
    default=False,
    help="Force reindex all documents.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=None,
    help="Token chunk size (default: 1200).",
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=None,
    help="Chunk overlap (default: 100).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model name (default: gpt-4o-mini).",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="OpenAI API key (default: OPENAI_API_KEY env var).",
)
@click.option(
    "--base-url",
    type=str,
    default=None,
    help="API base URL for custom endpoints.",
)
@click.option(
    "--graph-backend",
    type=click.Choice(["networkx", "neo4j"]),
    default=None,
    help="Graph backend (default: networkx).",
)
@click.option(
    "--project-id",
    type=str,
    default=None,
    help="Project ID for data isolation (default: default).",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="LLM timeout in seconds (default: 600).",
)
@click.pass_context
def add_documents(
    ctx: click.Context,
    sources: tuple[str, ...],
    incremental: bool | None,
    force_reindex: bool,
    chunk_size: int | None,
    chunk_overlap: int | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    graph_backend: str | None,
    project_id: str | None,
    timeout: int | None,
) -> None:
    """Add documents to the knowledge graph.

    SOURCES can be file paths, glob patterns (e.g., "docs/*.pdf"),
    or URLs (e.g., "https://example.com/page.html").

    Examples:

        manas add-documents document.pdf

        manas add-documents "docs/**/*.md" --model gpt-4o

        manas add-documents https://example.com/page.html

        manas add-documents file1.txt file2.txt --force-reindex
    """
    config = ctx.obj["config"]
    index_config = config.get("index", {})
    verbose = ctx.obj["verbose"]

    # Resolve configuration values
    working_dir = ctx.obj["working_dir"]
    chunk_size = _resolve_value(chunk_size, index_config.get("chunk_size"), default=1200)
    chunk_overlap = _resolve_value(chunk_overlap, index_config.get("chunk_overlap"), default=100)
    model = _resolve_value(model, config.get("model"), default="gpt-4o-mini")
    api_key = _resolve_value(api_key, config.get("api_key"), env_var="OPENAI_API_KEY")
    base_url = _resolve_value(base_url, config.get("base_url"), env_var="OPENAI_BASE_URL")
    graph_backend = _resolve_value(graph_backend, config.get("graph_backend"), default="networkx")
    incremental = _resolve_value(incremental, index_config.get("incremental"), default=True)
    timeout = _resolve_value(timeout, config.get("timeout"), env_var="MANAS_TIMEOUT", default=600)

    # Load documents
    loader = DocumentLoader(verbose=verbose)

    if verbose:
        click.echo(f"Loading documents from {len(sources)} source(s)...")

    try:
        documents = loader.load(list(sources))
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except ImportError as e:
        raise click.ClickException(str(e))

    if not documents:
        raise click.ClickException("No documents found to index.")

    click.echo(f"Loaded {len(documents)} document(s)")

    # Build ManasRAG instance
    if verbose:
        click.echo(f"Initializing ManasRAG (working_dir={working_dir}, model={model})...")

    try:
        manas = _build_manasrag(
            working_dir=working_dir,
            model=model,
            api_key=api_key,
            base_url=base_url,
            graph_backend=graph_backend,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            timeout=timeout,
            verbose=verbose,
        )
    except ManasRAGConfigError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Failed to initialize ManasRAG: {e}")

    # Index documents
    if verbose:
        click.echo("Indexing documents...")

    try:
        result = manas.index(
            documents=documents,
            project_id=project_id,
            incremental=incremental,
            force_reindex=force_reindex,
        )
    except Exception as e:
        raise click.ClickException(f"Indexing failed: {e}")

    # Print results
    click.echo("\nIndexing complete!")
    click.echo(f"  Status: {result.get('status', 'unknown')}")

    if "documents_count" in result:
        click.echo(f"  Documents: {result['documents_count']}")
    if "new_documents" in result:
        click.echo(f"  New documents: {result['new_documents']}")
    if "chunks_count" in result:
        click.echo(f"  Chunks: {result['chunks_count']}")
    if "new_chunks" in result:
        click.echo(f"  New chunks: {result['new_chunks']}")
    if "entities_count" in result:
        click.echo(f"  Entities: {result['entities_count']}")
    if "relations_count" in result:
        click.echo(f"  Relations: {result['relations_count']}")
    if "communities_count" in result:
        click.echo(f"  Communities: {result['communities_count']}")


@cli.command()
@click.argument("query", required=False)
@click.option(
    "-m", "--mode",
    type=click.Choice(["naive", "local", "global", "bridge", "nobridge", "hi"]),
    default=None,
    help="Retrieval mode (default: hi).",
)
@click.option(
    "--top-k",
    type=int,
    default=None,
    help="Number of entities to retrieve (default: 20).",
)
@click.option(
    "--top-m",
    type=int,
    default=None,
    help="Key entities per community (default: 10).",
)
@click.option(
    "--response-type",
    type=str,
    default=None,
    help='Response format (default: "Multiple Paragraphs").',
)
@click.option(
    "--context-only",
    is_flag=True,
    default=False,
    help="Return only retrieved context, no LLM generation.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model name.",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="OpenAI API key.",
)
@click.option(
    "--base-url",
    type=str,
    default=None,
    help="API base URL.",
)
@click.option(
    "--graph-backend",
    type=click.Choice(["networkx", "neo4j"]),
    default=None,
    help="Graph backend.",
)
@click.option(
    "--stdin",
    is_flag=True,
    default=False,
    help="Read query from stdin.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output result as JSON.",
)
@click.option(
    "--project-id",
    type=str,
    default=None,
    help="Project ID for data isolation (default: default).",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="LLM timeout in seconds (default: 600).",
)
@click.pass_context
def query(
    ctx: click.Context,
    query: str | None,
    mode: str | None,
    top_k: int | None,
    top_m: int | None,
    response_type: str | None,
    context_only: bool,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    graph_backend: str | None,
    stdin: bool,
    output_json: bool,
    project_id: str | None,
    timeout: int | None,
) -> None:
    """Query the knowledge graph.

    QUERY is the question to ask. Can also be provided via --stdin.

    Examples:

        manas query "What are the main themes?"

        manas query "Summarize the key concepts" --mode hi_global

        echo "What is AI?" | manas query --stdin

        manas query "List entities" --context-only --json
    """
    config = ctx.obj["config"]
    query_config = config.get("query", {})
    verbose = ctx.obj["verbose"]

    # Get query text
    if stdin:
        query_text = sys.stdin.read().strip()
    elif query:
        query_text = query
    else:
        raise click.ClickException("Query is required. Provide as argument or use --stdin.")

    if not query_text:
        raise click.ClickException("Empty query provided.")

    # Resolve configuration values
    working_dir = ctx.obj["working_dir"]
    mode = _resolve_value(mode, query_config.get("mode"), default="hi")
    top_k = _resolve_value(top_k, query_config.get("top_k"), default=20)
    top_m = _resolve_value(top_m, query_config.get("top_m"), default=10)
    response_type = _resolve_value(
        response_type,
        query_config.get("response_type"),
        default="Multiple Paragraphs",
    )
    model = _resolve_value(model, config.get("model"), default="gpt-4o-mini")
    api_key = _resolve_value(api_key, config.get("api_key"), env_var="OPENAI_API_KEY")
    base_url = _resolve_value(base_url, config.get("base_url"), env_var="OPENAI_BASE_URL")
    graph_backend = _resolve_value(graph_backend, config.get("graph_backend"), default="networkx")
    timeout = _resolve_value(timeout, config.get("timeout"), env_var="MANAS_TIMEOUT", default=600)

    # Build ManasRAG instance
    if verbose:
        click.echo(f"Initializing ManasRAG (working_dir={working_dir})...")

    try:
        manas = _build_manasrag(
            working_dir=working_dir,
            model=model,
            api_key=api_key,
            base_url=base_url,
            graph_backend=graph_backend,
            chunk_size=1200,  # Not used for query
            chunk_overlap=100,
            timeout=timeout,
            top_k=top_k,
            top_m=top_m,
            verbose=verbose,
        )
    except ManasRAGConfigError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Failed to initialize ManasRAG: {e}")

    # Build query parameters
    param = QueryParam(
        mode=RetrievalMode(mode),
        top_k=top_k,
        top_m=top_m,
        response_type=response_type,
        only_need_context=context_only,
    )

    if verbose:
        click.echo(f"Querying with mode={mode}, top_k={top_k}, top_m={top_m}...")

    # Execute query
    try:
        result = manas.query(query=query_text, mode=mode, param=param, project_id=project_id)
    except Exception as e:
        raise click.ClickException(f"Query failed: {e}")

    # Output result
    if output_json:
        # Convert to JSON-serializable format
        output = {
            "query": query_text,
            "mode": mode,
            "answer": result.get("answer", ""),
            "context": result.get("context", ""),
        }
        click.echo(json.dumps(output, indent=2))
    else:
        if context_only:
            click.echo("\n--- Context ---\n")
            click.echo(result.get("context", "No context retrieved."))
        else:
            click.echo("\n--- Answer ---\n")
            click.echo(result.get("answer", "No answer generated."))

            if verbose and result.get("context"):
                click.echo("\n--- Context ---\n")
                click.echo(result.get("context", ""))


@cli.command()
@click.option(
    "--host",
    type=str,
    default=None,
    help="Bind host (default: 0.0.0.0).",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Bind port (default: 8000).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model name.",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="OpenAI API key.",
)
@click.option(
    "--base-url",
    type=str,
    default=None,
    help="API base URL for custom endpoints.",
)
@click.option(
    "--graph-backend",
    type=click.Choice(["networkx", "neo4j"]),
    default=None,
    help="Graph backend (default: networkx).",
)
@click.option(
    "--reload/--no-reload",
    default=False,
    help="Enable auto-reload for development.",
)
@click.pass_context
def serve(
    ctx: click.Context,
    host: str | None,
    port: int | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    graph_backend: str | None,
    reload: bool,
) -> None:
    """Start the ManasRAG REST API server.

    Provides both native ManasRAG endpoints (/api/*) and OpenAI-compatible
    chat completions (/v1/*) for integration with tools like Open WebUI.

    Examples:

        manas serve

        manas serve --port 8080 --model gpt-4o

        manas serve --reload  # Development mode with auto-reload

    Config file support (manas.yaml):

        server:
          host: "0.0.0.0"
          port: 8000

    Environment variables:
        MANAS_HOST, MANAS_PORT
    """
    # Check for uvicorn
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "API dependencies not installed. Run: pip install manas[api]"
        )

    config = ctx.obj["config"]
    server_config = config.get("server", {})
    verbose = ctx.obj["verbose"]

    # Resolve configuration values
    working_dir = ctx.obj["working_dir"]
    host = _resolve_value(host, server_config.get("host"), env_var="MANAS_HOST", default="0.0.0.0")
    port = _resolve_value(port, server_config.get("port"), env_var="MANAS_PORT", default=8000)
    model = _resolve_value(model, config.get("model"), default="gpt-4o-mini")
    api_key = _resolve_value(api_key, config.get("api_key"), env_var="OPENAI_API_KEY")
    base_url = _resolve_value(base_url, config.get("base_url"), env_var="OPENAI_BASE_URL")
    graph_backend = _resolve_value(graph_backend, config.get("graph_backend"), default="networkx")

    # Handle port as int (could come from env var as string)
    if isinstance(port, str):
        port = int(port)

    # Print startup info
    click.echo("Starting ManasRAG API server...")
    click.echo(f"  Working directory: {working_dir}")
    click.echo(f"  Model: {model}")
    click.echo(f"  Graph backend: {graph_backend}")
    click.echo(f"  Host: {host}")
    click.echo(f"  Port: {port}")
    click.echo()
    click.echo("Endpoints:")
    click.echo(f"  Native API:     http://{host}:{port}/api/")
    click.echo(f"  OpenAI API:     http://{host}:{port}/v1/")
    click.echo(f"  Documentation:  http://{host}:{port}/docs")
    click.echo()

    # Import and configure the app
    from manasrag.api import AppConfig, create_app

    app_config = AppConfig(
        working_dir=working_dir,
        model=model,
        api_key=api_key,
        base_url=base_url,
        graph_backend=graph_backend,
    )

    if reload:
        # For reload mode, use uvicorn's factory mode
        # Configure via environment variables for the reloaded process
        click.echo("Running in development mode with auto-reload...")
        click.echo("Note: Config changes require restart.")

        # Store config in environment for the reloaded process
        os.environ["_MANAS_WORKING_DIR"] = working_dir
        os.environ["_MANAS_MODEL"] = model or ""
        os.environ["_MANAS_GRAPH_BACKEND"] = graph_backend
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url

        uvicorn.run(
            "manasrag.api.app:_create_dev_app",
            host=host,
            port=port,
            reload=True,
            reload_dirs=[str(Path(__file__).parent)],
            factory=True,
        )
    else:
        # Create app directly
        app = create_app(config=app_config)
        uvicorn.run(app, host=host, port=port)


@cli.command()
@click.option(
    "--format",
    type=click.Choice(["yaml", "json", "env"]),
    default="yaml",
    help="Output format (default: yaml).",
)
def default_config(format: str) -> None:
    """Output the default configuration for ManasRAG.

    This command prints a complete configuration file that can be
    saved and customized for your setup.

    Examples:

        manas default-config > manas.yaml

        manas default-config --format json

        manas default-config --format env > .env.example
    """
    config = {
        "working_dir": "./manas_data",
        "model": "gpt-4o-mini",
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": None,  # Optional, for custom endpoints
        "graph_backend": "networkx",
        "index": {
            "chunk_size": 1200,
            "chunk_overlap": 100,
            "incremental": True,
        },
        "query": {
            "mode": "hi",
            "top_k": 20,
            "top_m": 10,
            "response_type": "Multiple Paragraphs",
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
        },
    }

    env_vars = [
        ("OPENAI_API_KEY", "Your OpenAI API key"),
        ("OPENAI_BASE_URL", "Optional: Custom API endpoint base URL"),
        ("MANAS_WORKING_DIR", "Working directory (default: ./manas_data)"),
        ("MANAS_HOST", "Server bind host (default: 0.0.0.0)"),
        ("MANAS_PORT", "Server bind port (default: 8000)"),
    ]

    if format == "json":
        click.echo(json.dumps(config, indent=2))
    elif format == "env":
        for var, desc in env_vars:
            click.echo(f"# {desc}")
            if var == "OPENAI_API_KEY":
                click.echo(f"{var}=your_api_key_here")
            elif var == "MANAS_PORT":
                click.echo(f"{var}=8000")
            else:
                click.echo(f"# {var}=")
            click.echo()
    else:
        # YAML format with comments
        output = """# ManasRAG Configuration File
# Copy this to manas.yaml or ~/.manas.yaml

# Working directory for cache and data
working_dir: ./manas_data

# LLM model (see https://platform.openai.com/docs/models)
model: gpt-4o-mini

# OpenAI API key (can also use OPENAI_API_KEY env var)
# api_key: sk-...

# Optional: Custom API endpoint base URL
# base_url: https://api.openai.com/v1

# Graph backend: networkx (in-memory) or neo4j (production)
graph_backend: networkx

# Indexing configuration
index:
  # Token chunk size for document splitting
  chunk_size: 1200
  # Overlap between chunks
  chunk_overlap: 100
  # Skip already-indexed documents
  incremental: true

# Query configuration
query:
  # Retrieval mode: naive, hi_local, hi_global, hi_bridge, hi_nobridge, hi
  mode: hi
  # Number of entities to retrieve
  top_k: 20
  # Key entities per community for path finding
  top_m: 10
  # Response format
  response_type: Multiple Paragraphs

# REST API server configuration
server:
  # Bind address
  host: 0.0.0.0
  # Bind port
  port: 8000
"""
        click.echo(output)


@cli.command(name="visualize")
@click.option(
    "--type",
    "viz_type",
    type=click.Choice(["graph", "communities", "stats", "path", "all"]),
    default="all",
    help="Type of visualization to generate (default: all).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for visualizations (default: ./manas_visualizations).",
)
@click.option(
    "--layout",
    type=click.Choice(["force", "hierarchical", "circular"]),
    default="force",
    help="Layout algorithm for graph visualization (default: force).",
)
@click.option(
    "--color-by",
    type=click.Choice(["entity_type", "community", "degree"]),
    default="entity_type",
    help="How to color nodes (default: entity_type).",
)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    help="Open visualization in browser (default: true).",
)
@click.option(
    "--graph-backend",
    type=click.Choice(["networkx", "neo4j"]),
    default=None,
    help="Graph backend (default: networkx).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model name (only needed for path visualization).",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="OpenAI API key (only needed for path visualization).",
)
@click.option(
    "--base-url",
    type=str,
    default=None,
    help="API base URL for custom endpoints.",
)
@click.option(
    "--project-id",
    type=str,
    default=None,
    help="Project ID for data isolation (default: default).",
)
@click.pass_context
def visualize(
    ctx: click.Context,
    viz_type: str,
    output_dir: str | None,
    layout: str,
    color_by: str,
    open_browser: bool,
    graph_backend: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    project_id: str | None,
) -> None:
    """Generate and open interactive knowledge graph visualizations.

    Generates HTML-based visualizations that can be viewed in a web browser.
    Supports knowledge graph structure, community clustering, entity statistics,
    and query path visualizations.

    Examples:

        manas visualize

        manas visualize --type graph --layout hierarchical

        manas visualize --type communities --open false

        manas visualize --type stats --output-dir ./my_viz

        manas visualize --type path --model gpt-4o
    """
    import platform

    config = ctx.obj["config"]
    verbose = ctx.obj["verbose"]

    # Resolve configuration values
    working_dir = ctx.obj["working_dir"]
    graph_backend = _resolve_value(graph_backend, config.get("graph_backend"), default="networkx")
    model = _resolve_value(model, config.get("model"), default="gpt-4o-mini")
    api_key = _resolve_value(api_key, config.get("api_key"), env_var="OPENAI_API_KEY")
    base_url = _resolve_value(base_url, config.get("base_url"), env_var="OPENAI_BASE_URL")

    # Check if data exists
    if not Path(working_dir).exists():
        raise click.ClickException(
            f"No data found at {working_dir}. "
            "Please run 'manas add-documents' first to index some documents."
        )

    # Resolve output directory
    if output_dir is None:
        output_dir = "./manas_visualizations"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check visualization dependencies
    try:
        from pyvis.network import Network
    except ImportError:
        raise click.ClickException(
            "Visualization dependencies not installed. Run: pip install manas[visualization]"
        )

    if verbose:
        click.echo(f"Initializing ManasRAG (working_dir={working_dir}, backend={graph_backend})...")

    # Initialize ManasRAG (minimal, just for stores)
    try:
        manas = _build_manasrag(
            working_dir=working_dir,
            model=model,
            api_key=api_key,
            base_url=base_url,
            graph_backend=graph_backend,
            chunk_size=1200,
            chunk_overlap=100,
            timeout=ctx.obj.get("timeout", 600),
            verbose=verbose,
        )
    except ManasRAGConfigError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Failed to initialize ManasRAG: {e}")

    # Create visualizer
    visualizer = GraphVisualizer(
        output_dir=str(output_path),
        default_layout=layout,
    )

    generated_files: dict[str, str] = {}

    try:
        # Get project-specific graph store
        effective_project_id = project_id or "default"
        project_graph_store = manas.get_graph_store(project_id=effective_project_id)
        project_communities = project_graph_store._communities if hasattr(project_graph_store, "_communities") else {}

        if viz_type in ("graph", "all"):
            click.echo("Generating knowledge graph visualization...")
            html_path = visualizer.visualize_knowledge_graph(
                graph_store=project_graph_store,
                layout=layout,
                color_by=color_by,
                show_labels=True,
                physics=True,
            )
            generated_files["knowledge_graph"] = html_path
            click.echo(f"  -> {html_path}")

        if viz_type in ("communities", "all"):
            click.echo("Generating community visualization...")
            # Detect communities if needed
            if not project_communities:
                if verbose:
                    click.echo("Detecting communities...")
                project_communities = project_graph_store.clustering()
                project_graph_store._communities = project_communities

            html_path = visualizer.visualize_communities(
                communities=project_communities,
                graph_store=project_graph_store,
                layout=layout,
                show_community_labels=True,
                show_entity_labels=False,
            )
            generated_files["communities"] = html_path
            click.echo(f"  -> {html_path}")

        if viz_type in ("stats", "all"):
            click.echo("Generating entity statistics visualization...")
            html_path = visualizer.visualize_entity_stats(
                graph_store=project_graph_store,
                chart_types=["distribution", "degree"],
                show_top_n=20,
            )
            generated_files["statistics"] = html_path
            click.echo(f"  -> {html_path}")

        if viz_type == "path":
            click.echo("Query path visualization requires source and target entities.")
            click.echo("Use: manas visualize --type path --help for details")
            return

        # Summary
        click.echo(f"\nGenerated {len(generated_files)} visualization(s)")

        # Open in browser
        if open_browser and generated_files:
            # Prefer knowledge graph or first available
            primary_file = generated_files.get("knowledge_graph") or list(generated_files.values())[0]
            file_path = str(Path(primary_file).resolve())

            # Use file:// URL with proper path format
            if platform.system() == "Windows":
                file_url = f"file:///{file_path}"
            else:
                file_url = f"file://{file_path}"

            click.echo(f"\nOpening {Path(primary_file).name} in browser...")
            webbrowser.open(file_url)

    except Exception as e:
        raise click.ClickException(f"Visualization failed: {e}")


@cli.command(name="visualize-path")
@click.argument("source", required=True)
@click.argument("target", required=True)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for visualizations (default: ./manas_visualizations).",
)
@click.option(
    "--show-context",
    type=int,
    default=1,
    help="Number of neighboring nodes to show (default: 1).",
)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    help="Open visualization in browser (default: true).",
)
@click.option(
    "--graph-backend",
    type=click.Choice(["networkx", "neo4j"]),
    default=None,
    help="Graph backend (default: networkx).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model name.",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="OpenAI API key.",
)
@click.option(
    "--base-url",
    type=str,
    default=None,
    help="API base URL for custom endpoints.",
)
@click.option(
    "--project-id",
    type=str,
    default=None,
    help="Project ID for data isolation (default: default).",
)
@click.pass_context
def visualize_path(
    ctx: click.Context,
    source: str,
    target: str,
    output_dir: str | None,
    show_context: int,
    open_browser: bool,
    graph_backend: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    project_id: str | None,
) -> None:
    """Visualize the shortest path between two entities.

    Finds and visualizes the connection path between SOURCE entity and TARGET entity
    through the knowledge graph.

    Examples:

        manas visualize-path "AI" "Knowledge Graphs"

        manas visualize-path "Neural Networks" "Transformers" --show-context 2

        manas visualize-path "GPT" "LLM" --no-open --output-dir ./my_viz
    """
    import platform

    config = ctx.obj["config"]
    verbose = ctx.obj["verbose"]

    # Resolve configuration values
    working_dir = ctx.obj["working_dir"]
    graph_backend = _resolve_value(graph_backend, config.get("graph_backend"), default="networkx")
    model = _resolve_value(model, config.get("model"), default="gpt-4o-mini")
    api_key = _resolve_value(api_key, config.get("api_key"), env_var="OPENAI_API_KEY")
    base_url = _resolve_value(base_url, config.get("base_url"), env_var="OPENAI_BASE_URL")

    # Check if data exists
    if not Path(working_dir).exists():
        raise click.ClickException(
            f"No data found at {working_dir}. "
            "Please run 'manas add-documents' first to index some documents."
        )

    # Resolve output directory
    if output_dir is None:
        output_dir = "./manas_visualizations"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check visualization dependencies
    try:
        from pyvis.network import Network
    except ImportError:
        raise click.ClickException(
            "Visualization dependencies not installed. Run: pip install manas[visualization]"
        )

    if verbose:
        click.echo(f"Initializing ManasRAG (working_dir={working_dir})...")

    # Initialize ManasRAG
    try:
        manas = _build_manasrag(
            working_dir=working_dir,
            model=model,
            api_key=api_key,
            base_url=base_url,
            graph_backend=graph_backend,
            chunk_size=1200,
            chunk_overlap=100,
            timeout=ctx.obj.get("timeout", 600),
            verbose=verbose,
        )
    except ManasRAGConfigError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Failed to initialize ManasRAG: {e}")

    # Create visualizer
    visualizer = GraphVisualizer(
        output_dir=str(output_path),
        default_layout="hierarchical",
    )

    try:
        # Get project-specific graph store
        effective_project_id = project_id or "default"
        project_graph_store = manas.get_graph_store(project_id=effective_project_id)

        # Find path
        click.echo(f"Finding path from '{source}' to '{target}' in project '{effective_project_id}'...")
        path = project_graph_store.shortest_path(source.upper(), target.upper())

        if not path or len(path) < 2:
            raise click.ClickException(
                f"No path found between '{source}' and '{target}'. "
                "Try different entity names or check if entities exist."
            )

        click.echo(f"Path found: {' -> '.join(path)}")

        # Generate visualization
        html_path = visualizer.visualize_query_path(
            path=path,
            graph_store=project_graph_store,
            show_context=show_context,
            animate=True,
            include_relation_descriptions=True,
        )

        click.echo(f"Visualization saved to: {html_path}")

        # Open in browser
        if open_browser:
            file_path = str(Path(html_path).resolve())
            if platform.system() == "Windows":
                file_url = f"file:///{file_path}"
            else:
                file_url = f"file://{file_path}"

            click.echo("Opening in browser...")
            webbrowser.open(file_url)

    except Exception as e:
        raise click.ClickException(f"Path visualization failed: {e}")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
