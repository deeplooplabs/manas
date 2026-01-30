"""ManasRAG REST API package.

This package provides a REST API server for ManasRAG with:
- Native ManasRAG endpoints (/api/*)
- OpenAI-compatible chat completions (/v1/*)

Usage:
    ```python
    from manasrag.api import create_app, AppConfig

    config = AppConfig(
        working_dir="./manas_data",
        model="gpt-4o-mini",
        api_key="sk-...",
    )
    app = create_app(config=config)

    # Run with uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    ```

Or via CLI:
    ```bash
    manas serve --port 8000
    ```
"""

from manasrag.api.app import AppConfig, create_app

__all__ = ["create_app", "AppConfig"]
