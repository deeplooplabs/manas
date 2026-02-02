# ManasRAG WebUI

A user-friendly Streamlit interface for the ManasRAG hierarchical retrieval system.

## Features

- **Document Indexing**: Upload files (PDF, TXT, MD, DOCX, HTML, CSV), paste text, or load from directory
- **Query Interface**: Ask questions using multiple retrieval modes (naive, local, global, bridge, hi)
- **Visualization**: Interactive knowledge graph and community structure visualization
- **Statistics**: View detailed statistics about your knowledge graph
- **Project Management**: Multi-project support for data isolation

## Installation

Install the webui extra dependencies:

```bash
uv sync --extra webui
# or
pip install streamlit>=1.28.0
```

## Configuration

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
OPENAI_MODEL=gpt-4o-mini  # Optional
```

Or configure these directly in the WebUI sidebar.

## Usage

Start the WebUI:

```bash
uv run streamlit run webui/app.py
```

Or using the installed module:

```bash
streamlit run webui/app.py
```

The UI will open in your browser at `http://localhost:8501`

## Tabs

### 1. Indexing Tab

- **Upload Files**: Select and upload documents from your computer
- **Paste Text**: Directly paste text content to index
- **Load from Directory**: Specify a directory path with glob pattern

### 2. Query Tab

- **Retrieval Modes**:
  - `naive`: Basic RAG with document chunks
  - `local`: Entity-level retrieval with relations
  - `global`: Community report-level retrieval
  - `bridge`: Cross-community reasoning paths
  - `nobridge`: Local + global without paths (faster)
  - `hi`: Full hierarchical (all of the above)

- **Advanced Parameters**: Configure top-k, top-m, community level, response type

### 3. Visualization Tab

- **Knowledge Graph**: Interactive graph visualization
- **Communities**: Community structure visualization
- **Entity Statistics**: Entity-level statistics

### 4. Statistics Tab

View detailed statistics about:
- Graph nodes and edges
- Community distribution
- Indexed documents

## Tips

1. First time indexing may take a while as it extracts entities and builds the knowledge graph
2. Use "local" mode for entity-specific questions
3. Use "global" mode for high-level topic questions
4. Use "bridge" or "hi" mode for complex reasoning across topics
5. Adjust top-k and top-m parameters to balance speed and accuracy
