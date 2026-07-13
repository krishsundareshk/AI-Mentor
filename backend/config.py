import os
import re
from pathlib import Path

# --------------------------------------------------------------------------
# Storage location
# --------------------------------------------------------------------------
# Everything the app writes (SQLite memory DB, exported PDFs, vector DBs,
# topic checklists) lives under K:\AI-Mentor by default, never on C:.
# Override with the AI_MENTOR_DATA_DIR env var if you ever need to (e.g. for
# testing on a machine without a K: drive).
BASE_DIR = Path(os.environ.get("AI_MENTOR_DATA_DIR", "K:/AI-Mentor"))
DATA_DIR = BASE_DIR / "data"
EXPORTS_DIR = BASE_DIR / "exports"
DB_PATH = DATA_DIR / "memory.db"

# --------------------------------------------------------------------------
# Subject ebook library
# --------------------------------------------------------------------------
# Drop your ebooks (PDF) into Books/<Subject>/*.pdf -- one subfolder per
# subject below, as many books per subject as you like. Each subject gets
# its own isolated vector database (own PersistentClient directory), and its
# own topic checklist, so subjects never bleed into each other.
BOOKS_DIR = BASE_DIR / "Books"

# The canonical subject list (folder names must match exactly, case-sensitive
# is not required -- matching is case-insensitive, see subject_slug below).
SUBJECTS = [
    "Python",
    "Git",
    "DSA",
    "OS",
    "CN",
    "DBMS",
    "Software Engineering",
    "System Design",
    "Data Science",
    "ML",
    "DL",
    "AI & LLMs",
    "MLOps",
    "DevOps",
    "Cloud",
    "Data Engineering",
    "Interview Preparation",
]

# Vector DBs: one isolated PersistentClient directory per subject, under
# vector_db/subjects/<slug>. Workspace (your own project code) keeps its own
# separate collection, as before.
CHROMA_DIR = BASE_DIR / "vector_db"
SUBJECTS_CHROMA_DIR = CHROMA_DIR / "subjects"

# Per-subject topic checklists (JSON), so "done" status survives re-ingestion.
TOPICS_DIR = DATA_DIR / "topics"

WORKSPACE_CONFIG_PATH = DATA_DIR / "workspace_config.json"
WORKSPACE_COLLECTION_NAME = "workspace"


def subject_slug(subject: str) -> str:
    """Filesystem/collection-safe key for a subject name, e.g.
    'AI & LLMs' -> 'ai_llms', 'System Design' -> 'system_design'."""
    slug = subject.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug


_SLUG_TO_SUBJECT = {subject_slug(s): s for s in SUBJECTS}


def subject_from_slug(slug: str) -> str | None:
    return _SLUG_TO_SUBJECT.get(slug)


def subject_books_dir(subject: str) -> Path:
    return BOOKS_DIR / subject


def subject_chroma_dir(subject: str) -> Path:
    return SUBJECTS_CHROMA_DIR / subject_slug(subject)


def subject_topics_path(subject: str) -> Path:
    return TOPICS_DIR / f"{subject_slug(subject)}.json"


# --------------------------------------------------------------------------
# Ollama / model config
# --------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen2.5-coder:7b")
EXPLAINER_MODEL = os.environ.get("EXPLAINER_MODEL", "qwen3:8b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# Generous timeout: on 8GB VRAM, Ollama will unload/reload models between
# Qwen and DeepSeek calls, which adds real wall-clock time.
MODEL_TIMEOUT_SECONDS = int(os.environ.get("MODEL_TIMEOUT_SECONDS", "600"))

# How many retrieved chunks to feed into the model calls per question.
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "5"))

# Force how many model layers Ollama offloads to the GPU. On an 8GB RTX 4060,
# both qwen2.5-coder:7b and deepseek-r1:8b (Q4 quant) fit entirely on-GPU one
# at a time, but Ollama doesn't always offload every layer by default if it's
# being conservative about VRAM headroom. Set OLLAMA_NUM_GPU=99 (or any
# number >= the model's layer count) to force full GPU offload; leave unset
# (None) to let Ollama decide automatically.
# To check whether the GPU is actually being used: run `ollama ps` while a
# request is in flight -- the PROCESSOR column should read "100% GPU", not
# "100% CPU" or a split like "43%/57% CPU/GPU". Also watch `nvidia-smi` for
# GPU utilization and VRAM usage while a call runs.
_num_gpu_env = os.environ.get("OLLAMA_NUM_GPU")
OLLAMA_NUM_GPU = int(_num_gpu_env) if _num_gpu_env else None


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    SUBJECTS_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    for subject in SUBJECTS:
        subject_books_dir(subject).mkdir(parents=True, exist_ok=True)
