"""Workspace Manager: index a local project folder so the mentor can give
code-aware explanations of YOUR existing code, not just generate new snippets.

Same Parse -> Clean -> Chunk -> Embed -> ChromaDB -> Retrieve pipeline as the
Knowledge Engine, but chunked by lines (code) instead of paragraphs (prose),
and scoped to a single "active" project root at a time.
"""
import json
from pathlib import Path

from config import CHROMA_DIR, WORKSPACE_CONFIG_PATH, WORKSPACE_COLLECTION_NAME, EMBED_MODEL, ensure_dirs
from ollama_client import embed_text, embed_texts
from chroma_client import get_collection
from ingest import _file_hash, _load_manifest, _save_manifest  # reuse shared helpers
from logger import get_logger

log = get_logger("workspace")

WORKSPACE_MANIFEST_PATH = CHROMA_DIR / "workspace_manifest.json"

CODE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".cc", ".c", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".cs", ".md", ".json", ".yaml", ".yml", ".sql",
}
IGNORE_DIR_NAMES = {
    "node_modules", ".git", "__pycache__", "venv", ".venv", "env", "dist",
    "build", ".next", "target", ".idea", ".vscode", "vector_db",
}
MAX_FILE_SIZE_BYTES = 500_000  # skip huge generated/lock files
LINES_PER_CHUNK = 80
LINE_OVERLAP = 15
EMBED_BATCH_SIZE = 64  # same batching win as ingest.py -- was one-at-a-time before


# --------------------------------------------------------------------------
# Active workspace root
# --------------------------------------------------------------------------
def set_workspace_root(path: str) -> dict:
    root = Path(path)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Path does not exist or is not a directory: {path}")
    ensure_dirs()
    WORKSPACE_CONFIG_PATH.write_text(json.dumps({"root": str(root)}), encoding="utf-8")
    return {"root": str(root)}


def get_workspace_root() -> Path | None:
    if not WORKSPACE_CONFIG_PATH.exists():
        return None
    data = json.loads(WORKSPACE_CONFIG_PATH.read_text(encoding="utf-8"))
    root = data.get("root")
    return Path(root) if root else None


# --------------------------------------------------------------------------
# Parse / discover files
# --------------------------------------------------------------------------
def _is_ignored(rel_path: Path) -> bool:
    for part in rel_path.parts[:-1]:  # exclude the filename itself
        if part in IGNORE_DIR_NAMES or part.startswith("."):
            return True
    return False


def _iter_code_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if _is_ignored(rel):
            continue
        if p.suffix.lower() not in CODE_EXTS:
            continue
        try:
            if p.stat().st_size > MAX_FILE_SIZE_BYTES:
                log.info(f"skipping {rel}: exceeds MAX_FILE_SIZE_BYTES ({MAX_FILE_SIZE_BYTES})")
                continue
        except OSError:
            continue
        yield p


# --------------------------------------------------------------------------
# Chunk (line-based, preserves line numbers for reference)
# --------------------------------------------------------------------------
def _chunk_code(text: str, lines_per_chunk: int = LINES_PER_CHUNK, overlap: int = LINE_OVERLAP) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []
    chunks = []
    i = 0
    n = len(lines)
    while i < n:
        block = lines[i:i + lines_per_chunk]
        start_line, end_line = i + 1, i + len(block)
        chunks.append(f"[lines {start_line}-{end_line}]\n" + "\n".join(block))
        if i + lines_per_chunk >= n:
            break
        i += max(lines_per_chunk - overlap, 1)
    return chunks


# --------------------------------------------------------------------------
# Embed + store (incremental, same hash-manifest pattern as the library)
# --------------------------------------------------------------------------
def ingest_workspace() -> dict:
    root = get_workspace_root()
    if root is None:
        raise ValueError("No workspace root set yet. Set one before indexing.")
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Workspace root no longer exists: {root}")

    ensure_dirs()
    manifest = _load_manifest(WORKSPACE_MANIFEST_PATH)
    collection = get_collection(WORKSPACE_COLLECTION_NAME)

    found_files = list(_iter_code_files(root))
    found_keys = set()

    stats = {
        "root": str(root),
        "files_scanned": len(found_files),
        "files_new": 0,
        "files_updated": 0,
        "files_skipped": 0,
        "files_removed": 0,
        "chunks_added": 0,
        "errors": [],
    }

    for path in found_files:
        rel_key = str(path.relative_to(root))
        found_keys.add(rel_key)

        try:
            file_hash = _file_hash(path)
        except OSError as e:
            stats["errors"].append(f"{rel_key}: could not read file ({e})")
            continue

        prior = manifest.get(rel_key)
        if prior and prior.get("hash") == file_hash:
            stats["files_skipped"] += 1
            continue

        if prior:
            old_ids = prior.get("chunk_ids", [])
            if old_ids:
                collection.delete(ids=old_ids)
            stats["files_updated"] += 1
        else:
            stats["files_new"] += 1

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            stats["errors"].append(f"{rel_key}: failed to read ({e})")
            continue

        chunks = _chunk_code(text)
        ids, embeddings, documents, metadatas = [], [], [], []
        batch_endpoint_broken = False  # falls back to one-at-a-time if /api/embed isn't supported
        for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
            batch_embeddings: list[list[float] | None] = [None] * len(batch)

            if not batch_endpoint_broken:
                try:
                    batch_embeddings = embed_texts(EMBED_MODEL, batch)
                except Exception as e:
                    log.warning(
                        f"{rel_key}: batch embedding failed ({e}), "
                        "falling back to one-at-a-time for the rest of this file"
                    )
                    batch_endpoint_broken = True
                    batch_embeddings = [None] * len(batch)

            if batch_endpoint_broken:
                for j, chunk in enumerate(batch):
                    try:
                        batch_embeddings[j] = embed_text(EMBED_MODEL, chunk)
                    except Exception as e:
                        i = batch_start + j
                        stats["errors"].append(f"{rel_key} chunk {i}: embedding failed ({e})")

            for j, (chunk, embedding) in enumerate(zip(batch, batch_embeddings)):
                if embedding is None:
                    continue
                i = batch_start + j
                ids.append(f"{rel_key}::chunk::{i}")
                embeddings.append(embedding)
                documents.append(chunk)
                metadatas.append({"source": rel_key, "chunk_index": i})

        if ids:
            collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

        manifest[rel_key] = {"hash": file_hash, "chunk_ids": ids}
        stats["chunks_added"] += len(ids)

    removed_keys = set(manifest.keys()) - found_keys
    for key in removed_keys:
        old_ids = manifest[key].get("chunk_ids", [])
        if old_ids:
            collection.delete(ids=old_ids)
        del manifest[key]
        stats["files_removed"] += 1

    _save_manifest(manifest, WORKSPACE_MANIFEST_PATH)
    return stats


def workspace_status() -> dict:
    root = get_workspace_root()
    manifest = _load_manifest(WORKSPACE_MANIFEST_PATH)
    files = [
        {"source": k, "chunks": len(v.get("chunk_ids", []))}
        for k, v in manifest.items()
    ]
    files.sort(key=lambda f: f["source"])
    return {
        "root": str(root) if root else None,
        "files": files,
        "total_files": len(files),
        "total_chunks": sum(f["chunks"] for f in files),
    }
