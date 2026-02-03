"""MinerU document converter for Haystack.

This module provides a Haystack component that uses MinerU to parse
PDF documents and images into structured Document objects.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

from haystack import component
from haystack.dataclasses import ByteStream, Document

# MinerU imports with graceful fallback
try:
    import mineru
    from mineru.cli.common import do_parse, read_fn
    from mineru.utils.enum_class import MakeMode
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
        union_make,
    )

    # Patch torch.load to allow weights_only=False for older model compatibility
    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False

logger = logging.getLogger(__name__)


@component
class MinerUToDocument:
    """Use MinerU to parse documents into Haystack Documents.

    MinerU is a powerful document intelligence system that supports PDF
    and image document parsing, extracting text, tables, formulas, layout
    structure, and outputting in Markdown format.

    Supported backends:
        - pipeline: Multi-model pipeline, suitable for general scenarios
        - vlm-auto-engine: Vision language model, local high-precision
        - vlm-http-client: Vision language model, remote high-precision
        - hybrid-auto-engine: Hybrid mode, local high-precision
        - hybrid-http-client: Hybrid mode, remote high-precision

    Example:
        ```python
        converter = MinerUToDocument(backend="hybrid-auto-engine")
        result = converter.run(sources=["document.pdf"])
        documents = result["documents"]
        ```
    """

    def __init__(
        self,
        backend: str = "hybrid-auto-engine",
        language: str = "ch",
        parse_method: str = "auto",
        formula_enable: bool = True,
        table_enable: bool = True,
        **kwargs,
    ):
        """Initialize the MinerU document converter.

        Args:
            backend: Parsing backend. Options: pipeline, vlm-auto-engine,
                vlm-http-client, hybrid-auto-engine, hybrid-http-client.
            language: Document language for OCR optimization.
            parse_method: Parsing method. "auto" for automatic detection,
                "txt" for text extraction, "ocr" for OCR recognition.
            formula_enable: Whether to enable formula recognition.
            table_enable: Whether to enable table recognition.

        Raises:
            ImportError: If MinerU is not installed.
        """
        if not MINERU_AVAILABLE:
            raise ImportError(
                "MinerU is not installed. "
                "Please install it with: pip install mineru"
            )

        self.backend = backend
        self.language = language
        self.parse_method = parse_method
        self.formula_enable = formula_enable
        self.table_enable = table_enable

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | None = None,
    ) -> dict[str, list[Document]]:
        """Execute document conversion.

        Args:
            sources: List of source file paths or ByteStream objects.
            meta: Optional metadata dictionary or list of metadata dicts.

        Returns:
            {"documents": [...]}
        """
        documents: list[Document] = []
        meta_list = self._normalize_metadata(meta, len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                if isinstance(source, ByteStream):
                    # Handle ByteStream from URL downloads
                    result_doc = self._parse_byte_stream(source, metadata)
                else:
                    # Handle file path
                    source_path = Path(source)
                    if not source_path.exists():
                        logger.warning(f"File not found: {source_path}")
                        continue
                    result_doc = self._parse_single_file(source_path)
                    if result_doc is not None:
                        result_doc.meta = {**result_doc.meta, **metadata}
                if result_doc is not None:
                    documents.append(result_doc)
            except Exception as e:
                logger.warning(f"Could not parse {source}: {e}")
                continue

        return {"documents": documents}

    def _parse_single_file(self, source_path: Path) -> Document | None:
        """Parse a single file using MinerU.

        Args:
            source_path: Path to the source file.

        Returns:
            A Document object or None if parsing failed.
        """
        pdf_bytes = read_fn(str(source_path))
        # Use only the file stem (name without extension) for MinerU
        file_name = source_path.stem
        return self._parse_bytes(pdf_bytes, file_name)

    def _parse_byte_stream(
        self, stream: ByteStream, metadata: dict[str, Any]
    ) -> Document | None:
        """Parse a ByteStream using MinerU.

        Args:
            stream: ByteStream object from URL downloads.
            metadata: Additional metadata to include.

        Returns:
            A Document object or None if parsing failed.
        """
        from urllib.parse import urlparse, unquote

        # Try to get URL from meta first, then from source_url attribute
        source_url = stream.meta.get("url") if stream.meta else None
        if not source_url:
            source_url = getattr(stream, "source_url", None)

        if source_url:
            # Extract filename from URL
            parsed = urlparse(source_url)
            path = unquote(parsed.path)
            file_name = Path(path).stem or "document"
        else:
            file_name = "document"

        pdf_bytes = stream.data
        return self._parse_bytes(pdf_bytes, file_name, metadata)

    def _parse_bytes(
        self, pdf_bytes: bytes, source_name: str, extra_meta: dict | None = None
    ) -> Document | None:
        """Parse bytes using MinerU.

        Args:
            pdf_bytes: Document content as bytes.
            source_name: Name or path of the source.
            extra_meta: Additional metadata to include.

        Returns:
            A Document object or None if parsing failed.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Parse with MinerU
            middle_json = self._parse_with_mineru(
                pdf_bytes, source_name, tmp_dir
            )

            # Extract markdown content
            pdf_info = middle_json.get("pdf_info", [middle_json])
            md_content = union_make(pdf_info, MakeMode.MM_MD, "")

            meta = {
                "source_file": source_name,
                "backend": self.backend,
                "language": self.language,
            }
            if extra_meta:
                meta.update(extra_meta)

            return Document(
                content=md_content,
                meta=meta,
            )

    def _parse_with_mineru(
        self, pdf_bytes: bytes, file_name: str, output_dir: str
    ) -> dict:
        """Parse document using MinerU backend.

        Args:
            pdf_bytes: Document content as bytes.
            file_name: Name of the source file.
            output_dir: Temporary output directory.

        Returns:
            Middle JSON structure from MinerU.
        """
        # Prepare arguments for do_parse
        pdf_file_names = [file_name]
        pdf_bytes_list = [pdf_bytes]
        p_lang_list = [self.language]

        # Determine backend type and map to MinerU's expected format
        backend = self.backend

        if backend == "pipeline":
            # Multi-model pipeline
            do_parse(
                output_dir=output_dir,
                pdf_file_names=pdf_file_names,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=p_lang_list,
                backend="pipeline",
                parse_method=self.parse_method,
                formula_enable=self.formula_enable,
                table_enable=self.table_enable,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=False,
                f_dump_middle_json=True,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
                f_make_md_mode=MakeMode.MM_MD,
            )
        elif backend.startswith("vlm-"):
            # VLM-based parsing
            engine = "http" if "http" in backend else "auto"
            vlm_backend = f"vlm-{engine}-engine"
            do_parse(
                output_dir=output_dir,
                pdf_file_names=pdf_file_names,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=p_lang_list,
                backend=vlm_backend,
                parse_method=self.parse_method,
                formula_enable=self.formula_enable,
                table_enable=self.table_enable,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=False,
                f_dump_middle_json=True,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
                f_make_md_mode=MakeMode.MM_MD,
            )
        elif backend.startswith("hybrid-"):
            # Hybrid parsing
            engine = "http" if "http" in backend else "auto"
            hybrid_backend = f"hybrid-{engine}-engine"
            do_parse(
                output_dir=output_dir,
                pdf_file_names=pdf_file_names,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=p_lang_list,
                backend=hybrid_backend,
                parse_method=self.parse_method,
                formula_enable=self.formula_enable,
                table_enable=self.table_enable,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=False,
                f_dump_middle_json=True,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
                f_make_md_mode=MakeMode.MM_MD,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        # Read the middle.json output (note: mineru uses {file_name}_middle.json)
        middle_json_path = os.path.join(
            output_dir, file_name, self.parse_method, f"{file_name}_middle.json"
        )
        with open(middle_json_path, "r", encoding="utf-8") as f:
            middle_json = json.load(f)

        return middle_json

    def _normalize_metadata(
        self, meta: dict[str, Any] | None, count: int
    ) -> list[dict[str, Any]]:
        """Normalize metadata to a list format.

        Args:
            meta: Metadata dict, list of dicts, or None.
            count: Number of expected metadata entries.

        Returns:
            List of metadata dictionaries.
        """
        if meta is None:
            return [{}] * count
        if isinstance(meta, dict):
            return [meta.copy()] * count
        if isinstance(meta, list):
            if len(meta) >= count:
                return meta[:count]
            return meta + [{}] * (count - len(meta))
        return [{}] * count
