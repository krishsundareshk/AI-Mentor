"""Retrieve relevant chunks for a question, either from one subject's own
vector db (Python, DSA, OS, ...) or from the workspace (your own project
code) collection."""
from config import EMBED_MODEL, RETRIEVAL_TOP_K
from ollama_client import embed_text
from chroma_client import get_collection, get_subject_collection


def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    subject: str | None = None,
    collection_name: str = "workspace",
) -> list[dict]:
    """Pass `subject` (e.g. "Python", "DSA") to search that subject's own
    vector db. Omit it to search a shared collection by name (currently
    only "workspace" exists)."""
    collection = get_subject_collection(subject) if subject else get_collection(collection_name)
    count = collection.count()
    if count == 0:
        return []

    query_embedding = embed_text(EMBED_MODEL, query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
    )

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            {
                "text": doc,
                "source": (meta or {}).get("source", "unknown"),
                "distance": dist,
            }
        )
    return out
