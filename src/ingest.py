"""
src/ingest.py

Load documents from the knowledge base, chunk them, embed, and save the index.
Run as:  python -m src.ingest
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running as a module from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import watsonx_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Document loaders
# ---------------------------------------------------------------------------

def _load_markdown(path: Path) -> tuple[str, dict]:
    """Load a .md or .txt file and extract front-matter metadata."""
    text = path.read_text(encoding="utf-8")
    meta = {"source_name": path.stem, "source_url": "", "file": path.name}

    # Parse simple YAML-like front matter
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if fm_match:
        for line in fm_match.group(1).splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                meta[key.strip()] = val.strip()
        text = text[fm_match.end():]

    return text, meta


def _load_pdf_pypdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_pdf(path: Path) -> str:
    """Try Docling first, fall back to pypdf."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(path))
        return result.document.export_to_markdown()
    except Exception as exc:
        logger.warning("Docling failed (%s), falling back to pypdf", exc)
        return _load_pdf_pypdf(path)


def _load_docx(path: Path) -> str:
    """Try Docling first, fall back to basic extraction."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(path))
        return result.document.export_to_markdown()
    except Exception as exc:
        logger.warning("Docling failed for docx (%s), falling back to basic read", exc)
        try:
            import zipfile, xml.etree.ElementTree as ET
            with zipfile.ZipFile(path) as z:
                xml_content = z.read("word/document.xml")
            root = ET.fromstring(xml_content)
            ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            return " ".join(t.text for t in root.iter(f"{ns}t") if t.text)
        except Exception:
            return ""


def load_document(path: Path) -> tuple[str, dict]:
    """Load any supported document type, return (text, meta)."""
    meta = {"source_name": path.stem, "source_url": "", "file": path.name}
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return _load_markdown(path)
    elif suffix == ".pdf":
        return _load_pdf(path), meta
    elif suffix == ".docx":
        return _load_docx(path), meta
    else:
        logger.warning("Unsupported file type: %s — skipping", path.name)
        return "", meta


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def chunk_text(text: str, meta: dict, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping chunks. Tries to split on paragraph/sentence
    boundaries to avoid breaking mid-sentence.
    """
    text = text.strip()
    if not text:
        return []

    # Normalise line endings
    text = re.sub(r"\r\n", "\n", text)

    chunks: list[dict] = []
    start = 0

    while start < len(text):
        end = start + size
        if end >= len(text):
            segment = text[start:]
        else:
            # Try to find a paragraph break
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + size // 2:
                end = para_break
            else:
                # Try a newline
                nl_break = text.rfind("\n", start, end)
                if nl_break > start + size // 2:
                    end = nl_break
                else:
                    # Try a sentence end (. ! ?)
                    sent_match = None
                    for m in re.finditer(r"[.!?]\s", text[start:end]):
                        sent_match = m
                    if sent_match and sent_match.end() + start > start + size // 2:
                        end = start + sent_match.end()
            segment = text[start:end]

        segment = segment.strip()
        if segment:
            chunks.append({
                "text": segment,
                "source_name": meta.get("source_name", ""),
                "source_url": meta.get("source_url", ""),
                "file": meta.get("file", ""),
            })

        # Advance with overlap
        next_start = end - overlap
        if next_start <= start:
            next_start = start + max(1, size - overlap)
        start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def save_index(chunks: list[dict], embeddings: np.ndarray):
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(config.INDEX_EMBEDDINGS_FILE), embeddings)
    with open(config.INDEX_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("Index saved: %d chunks", len(chunks))


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def run_ingestion(kb_dir: Optional[Path] = None) -> tuple[list[dict], np.ndarray]:
    kb_dir = kb_dir or config.KNOWLEDGE_BASE_DIR
    kb_dir = Path(kb_dir)

    supported = list(kb_dir.glob("*.md")) + list(kb_dir.glob("*.txt")) + \
                list(kb_dir.glob("*.pdf")) + list(kb_dir.glob("*.docx"))

    if not supported:
        print(f"ERROR: No supported documents found in {kb_dir}")
        print("Add .md / .txt / .pdf / .docx files to the knowledge_base folder.")
        sys.exit(1)

    t0 = time.time()
    all_chunks: list[dict] = []

    for path in sorted(supported):
        logger.info("Loading: %s", path.name)
        text, meta = load_document(path)
        if not text.strip():
            logger.warning("  -> empty, skipping")
            continue
        file_chunks = chunk_text(text, meta)
        logger.info("  -> %d chunks", len(file_chunks))
        all_chunks.extend(file_chunks)

    if not all_chunks:
        print("ERROR: All documents loaded as empty. Check your files.")
        sys.exit(1)

    print(f"\nEmbedding {len(all_chunks)} chunks from {len(supported)} files …")
    texts = [c["text"] for c in all_chunks]
    embeddings = watsonx_client.embed_texts(texts)

    save_index(all_chunks, embeddings)

    elapsed = time.time() - t0
    print(f"\nIngestion complete in {elapsed:.1f}s")
    print(f"  Files processed : {len(supported)}")
    print(f"  Chunks created  : {len(all_chunks)}")
    print(f"  Embedding shape : {embeddings.shape}")
    print(f"  Index saved to  : {config.INDEX_DIR}")

    return all_chunks, embeddings


if __name__ == "__main__":
    run_ingestion()
