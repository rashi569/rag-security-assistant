import os
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from security.prompt_injection_detector import scan_text_for_injection

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"

# all-MiniLM-L6-v2: small, fast, good-enough-quality embedding model that
# runs on CPU without issue - no GPU required for a project this size.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    return collection


def load_text_from_file(filepath: str) -> str:
    """Extracts raw text from a .pdf or .txt file."""
    path = Path(filepath)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Splits text into overlapping chunks by character count.

    Overlap matters: without it, a sentence that spans a chunk boundary
    gets cut in half and loses meaning for retrieval. A simple character-based
    splitter is used here (not sentence-aware) to keep this dependency-free -
    good enough for a portfolio project, real systems often use smarter
    sentence/paragraph-aware splitting.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def ingest_file(filepath: str, display_name: str = None) -> dict:
    """
    Ingests a single file: extracts text, chunks it, scans each chunk for
    prompt-injection patterns (a document could contain hidden instructions
    aimed at manipulating the LLM later - this is a real RAG attack vector
    called "indirect prompt injection"), embeds, and stores in Chroma.

    Returns a summary dict including any security flags raised during ingestion.
    """
    text = load_text_from_file(filepath)
    chunks = chunk_text(text)

    collection = get_chroma_collection()
    filename = display_name if display_name else os.path.basename(filepath)

    flagged_chunks = []
    ids, documents, metadatas = [], [], []

    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.sha256(f"{filename}-{i}-{chunk[:50]}".encode()).hexdigest()[:16]

        scan_result = scan_text_for_injection(chunk)
        if scan_result["flagged"]:
            flagged_chunks.append({"chunk_index": i, "reasons": scan_result["reasons"]})

        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "source": filename,
            "chunk_index": i,
            "injection_flagged": scan_result["flagged"],
        })

    if documents:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return {
        "filename": filename,
        "chunks_ingested": len(chunks),
        "flagged_chunks": flagged_chunks,
    }


def ingest_directory(directory: str) -> list[dict]:
    """Ingests every .pdf/.txt file in a directory."""
    results = []
    for filepath in Path(directory).glob("*"):
        if filepath.suffix.lower() in (".pdf", ".txt"):
            results.append(ingest_file(str(filepath)))
    return results


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "sample_docs"
    results = ingest_directory(target)
    for r in results:
        print(f"Ingested {r['filename']}: {r['chunks_ingested']} chunks")
        if r["flagged_chunks"]:
            print(f"  ⚠ {len(r['flagged_chunks'])} chunk(s) flagged for possible prompt injection")
