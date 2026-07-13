"""Persistent ChromaDB clients.

Two kinds of storage:
- The workspace collection (your own project code) -- one shared client, as
  before, at CHROMA_DIR.
- Subject collections (Python, DSA, OS, ...) -- each subject gets its OWN
  PersistentClient pointed at its own folder under vector_db/subjects/<slug>,
  so subjects are genuinely isolated vector databases, not just separate
  collections inside one shared index.
"""
import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, ensure_dirs, subject_chroma_dir

_client = None
_collections: dict = {}

_subject_clients: dict = {}
_subject_collections: dict = {}


def _get_client():
    global _client
    if _client is None:
        ensure_dirs()
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection(name: str = "workspace"):
    """Shared collection (currently just used for the workspace index)."""
    if name not in _collections:
        client = _get_client()
        _collections[name] = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[name]


def get_subject_collection(subject: str):
    """Own PersistentClient + collection for a single subject, e.g. 'Python'
    or 'DSA'. Each subject's vectors live entirely under their own folder on
    disk, independent of every other subject."""
    if subject not in _subject_collections:
        ensure_dirs()
        subj_dir = subject_chroma_dir(subject)
        subj_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(subj_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        _subject_clients[subject] = client
        _subject_collections[subject] = client.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"},
        )
    return _subject_collections[subject]
