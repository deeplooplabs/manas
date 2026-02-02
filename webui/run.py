"""Launch script for ManasRAG Streamlit WebUI."""

import subprocess
import sys


def main() -> None:
    """Launch the Streamlit WebUI."""
    print("Starting ManasRAG WebUI...")
    print("The interface will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "webui/app.py"],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
