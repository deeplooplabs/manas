"""Document loader for HiRAG CLI.

This module provides a DocumentLoader class that converts local files and URLs
to Haystack Document objects using the appropriate converters based on file type.
"""

import glob as glob_module
import hashlib
from pathlib import Path
from typing import Any

from haystack.dataclasses import Document, ByteStream


def _compute_doc_id(content: str) -> str:
    """Compute a deterministic document ID from content using MD5 hash.

    Args:
        content: Document content.

    Returns:
        MD5 hash hex string (32 characters).
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()


# ===== EXTENSION TO CONVERTER MAPPING =====
# Maps file extensions to (module_path, class_name) tuples for lazy import
EXTENSION_TO_CONVERTER: dict[str, tuple[str, str]] = {
    ".pdf": ("haystack.components.converters", "PyPDFToDocument"),
    ".docx": ("haystack.components.converters", "DOCXToDocument"),
    ".html": ("haystack.components.converters", "HTMLToDocument"),
    ".htm": ("haystack.components.converters", "HTMLToDocument"),
    ".md": ("haystack.components.converters", "MarkdownToDocument"),
    ".markdown": ("haystack.components.converters", "MarkdownToDocument"),
    ".xlsx": ("haystack.components.converters", "XLSXToDocument"),
    ".csv": ("haystack.components.converters", "CSVToDocument"),
    ".txt": ("haystack.components.converters", "TextFileToDocument"),
}

# ===== MIME TYPE TO EXTENSION MAPPING =====
MIME_TO_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}


def _import_converter(module_path: str, class_name: str) -> type:
    """Lazily import a converter class.

    Args:
        module_path: Full module path (e.g., "haystack.components.converters").
        class_name: Class name to import.

    Returns:
        The converter class.

    Raises:
        ImportError: If the required optional dependency is not installed.
    """
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _get_extension_from_url(url: str) -> str:
    """Extract file extension from URL path.

    Args:
        url: URL string.

    Returns:
        File extension (e.g., ".pdf") or ".txt" as fallback.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path
    if "." in path:
        ext = "." + path.rsplit(".", 1)[-1].lower()
        if ext in EXTENSION_TO_CONVERTER:
            return ext
    return ".txt"


class DocumentLoader:
    """Load documents from files and URLs.

    This class provides a unified interface for loading documents from various
    sources (local files, glob patterns, URLs) and converting them to Haystack
    Document objects.

    Converters are lazily imported to avoid requiring all optional dependencies.
    Only the converter needed for the file type being processed is imported.

    Example:
        ```python
        loader = DocumentLoader()

        # Load from files
        docs = loader.load(["doc.pdf", "notes.md"])

        # Load from URLs
        docs = loader.load(["https://example.com/page.html"])

        # Load from glob pattern
        docs = loader.load(["docs/*.pdf"])
        ```
    """

    def __init__(self, verbose: bool = False):
        """Initialize the document loader.

        Args:
            verbose: If True, print progress information.
        """
        self.verbose = verbose
        self._converter_cache: dict[str, Any] = {}

    def load(self, sources: list[str]) -> list[Document]:
        """Load documents from multiple sources.

        Args:
            sources: List of file paths, glob patterns, or URLs.

        Returns:
            List of Haystack Document objects.

        Raises:
            FileNotFoundError: If a local file does not exist.
            ImportError: If required converter dependency is not installed.
        """
        # Separate URLs from local paths
        urls: list[str] = []
        local_paths: list[Path] = []

        for source in sources:
            if source.startswith("http://") or source.startswith("https://"):
                urls.append(source)
            else:
                # Expand glob patterns
                expanded = glob_module.glob(source, recursive=True)
                if expanded:
                    for path_str in expanded:
                        path = Path(path_str)
                        if path.is_file():
                            local_paths.append(path)
                else:
                    # Not a glob pattern, treat as literal path
                    path = Path(source)
                    if path.exists() and path.is_file():
                        local_paths.append(path)
                    elif path.exists() and path.is_dir():
                        if self.verbose:
                            print(f"Skipping directory: {source}")
                    else:
                        raise FileNotFoundError(f"File not found: {source}")

        documents: list[Document] = []

        # Process local files
        if local_paths:
            documents.extend(self._load_local_files(local_paths))

        # Process URLs
        if urls:
            documents.extend(self._load_urls(urls))

        return documents

    def _load_local_files(self, paths: list[Path]) -> list[Document]:
        """Load documents from local files.

        Files are grouped by extension and processed with the appropriate
        converter. This is more efficient than processing files one by one.

        Args:
            paths: List of file paths.

        Returns:
            List of Document objects.
        """
        # Group files by extension
        files_by_ext: dict[str, list[Path]] = {}
        for path in paths:
            ext = path.suffix.lower()
            if ext not in files_by_ext:
                files_by_ext[ext] = []
            files_by_ext[ext].append(path)

        documents: list[Document] = []

        for ext, file_paths in files_by_ext.items():
            if self.verbose:
                print(f"Loading {len(file_paths)} {ext} file(s)...")

            converter = self._get_converter(ext)
            # All Haystack converters accept 'sources' parameter with file paths
            result = converter.run(sources=file_paths)
            docs = result.get("documents", [])

            # Set document ID from path hash and add source metadata
            for doc, path in zip(docs, file_paths):
                doc.id = _compute_doc_id(str(path))
                if doc.meta is None:
                    doc.meta = {}
                doc.meta["source_file"] = str(path)

            documents.extend(docs)

        return documents

    def _load_urls(self, urls: list[str]) -> list[Document]:
        """Load documents from URLs.

        Uses Haystack's LinkContentFetcher to download content, then routes
        to the appropriate converter based on MIME type or URL extension.

        Args:
            urls: List of URLs.

        Returns:
            List of Document objects.
        """
        try:
            from haystack.components.fetchers import LinkContentFetcher
        except ImportError as e:
            raise ImportError(
                "LinkContentFetcher not available. "
                "Please install haystack-ai>=2.6."
            ) from e

        if self.verbose:
            print(f"Fetching {len(urls)} URL(s)...")

        fetcher = LinkContentFetcher()
        result = fetcher.run(urls=urls)
        streams: list[ByteStream] = result.get("streams", [])

        if not streams:
            return []

        # Group streams by inferred file type, maintain stream to URL mapping
        streams_by_ext: dict[str, list[tuple[ByteStream, str]]] = {}

        for stream, url in zip(streams, urls):
            # Determine extension from MIME type or URL
            mime_type = stream.mime_type if stream.mime_type else ""
            ext = MIME_TO_EXT.get(mime_type)
            if not ext or mime_type == "application/octet-stream":
                ext = _get_extension_from_url(url)

            if ext not in streams_by_ext:
                streams_by_ext[ext] = []
            streams_by_ext[ext].append((stream, url))

        documents: list[Document] = []

        for ext, stream_url_pairs in streams_by_ext.items():
            if self.verbose:
                print(f"Converting {len(stream_url_pairs)} {ext} stream(s)...")

            converter = self._get_converter(ext)
            byte_streams = [pair[0] for pair in stream_url_pairs]
            # All Haystack converters accept ByteStream in their sources parameter
            result = converter.run(sources=byte_streams)
            docs = result.get("documents", [])

            # Set document ID from URL hash and add source URL metadata
            for doc, (stream, url) in zip(docs, stream_url_pairs):
                doc.id = _compute_doc_id(url)
                if doc.meta is None:
                    doc.meta = {}
                doc.meta["source_url"] = url

            documents.extend(docs)

        return documents

    def _get_converter(self, extension: str) -> Any:
        """Get or create a converter instance for the given extension.

        Args:
            extension: File extension (e.g., ".pdf").

        Returns:
            Converter instance.

        Raises:
            ImportError: If the required dependency is not installed.
            ValueError: If the extension is not supported.
        """
        # Normalize extension
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext

        # Check cache first
        if ext in self._converter_cache:
            return self._converter_cache[ext]

        # Get converter info
        if ext not in EXTENSION_TO_CONVERTER:
            # Fall back to text converter
            ext = ".txt"

        module_path, class_name = EXTENSION_TO_CONVERTER[ext]

        try:
            converter_class = _import_converter(module_path, class_name)
        except (ImportError, AttributeError) as e:
            # Provide helpful error message for missing optional dependencies
            dep_map = {
                ".pdf": "pypdf",
                ".docx": "python-docx",
                ".html": "boilerpy3",
                ".htm": "boilerpy3",
                ".md": "markdown-it-py",
                ".markdown": "markdown-it-py",
                ".xlsx": "openpyxl",
            }
            dep = dep_map.get(ext)
            if dep:
                raise ImportError(
                    f"Cannot load {ext} files. Please install: pip install {dep}"
                ) from e
            raise

        converter = converter_class()
        self._converter_cache[ext] = converter
        return converter

    @staticmethod
    def supported_extensions() -> list[str]:
        """Return list of supported file extensions.

        Returns:
            List of supported extensions (e.g., [".pdf", ".docx", ...]).
        """
        return list(EXTENSION_TO_CONVERTER.keys())
