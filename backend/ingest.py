"""Subject ebook ingestion pipeline: Parse -> Clean -> Chunk -> Embed -> Chroma.

Books live at Books/<Subject>/*.pdf (see config.SUBJECTS for the canonical
subject list). Each subject gets its own isolated vector database (its own
PersistentClient directory, via chroma_client.get_subject_collection) and its
own manifest.json + topic checklist, so subjects never mix.

Incremental, same pattern as before: files are hashed, unchanged files are
skipped on re-ingest, changed files are re-embedded, and deleted files have
their chunks removed.
"""
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader

from config import EMBED_MODEL, SUBJECTS, ensure_dirs, subject_books_dir, subject_chroma_dir
from ollama_client import embed_text, embed_texts
from chroma_client import get_subject_collection
from topics import extract_topics_from_book, merge_subject_topics
from logger import get_logger

log = get_logger("ingest")

CHUNK_SIZE_CHARS = 1500
# Texts per /api/embed call -- the main ingestion speed lever. nomic-embed-text
# is small (~275MB), so on an 8GB card you have plenty of headroom to push
# this higher than 32 -- try 64 or 128 and watch VRAM/latency with `nvidia-smi`
# and `ollama ps` to find the sweet spot for your GPU.
EMBED_BATCH_SIZE = 64
CHUNK_OVERLAP_PARAGRAPHS = 1
SUPPORTED_EXTS = {".pdf"}  # ebooks only -- no crawled docs / markdown corpus anymore

# How long to keep the embedding model loaded between batches during a long
# ingestion run (a full subject can take hours). Without this, Ollama's
# default idle-unload window can kick in between books and force a reload.
EMBED_KEEP_ALIVE_DURING_INGEST = "30m"

# --------------------------------------------------------------------------
# Repeated header/footer stripping
# --------------------------------------------------------------------------
# PDF text extraction commonly repeats the exact same running header, footer,
# or page-number line on every page. Left in, this junk (a) pollutes chunks
# fed to the embedding model with noise unrelated to the actual content, and
# (b) is especially damaging to topic extraction, which only ever samples a
# handful of pages -- a repeated header can crowd out real content in a small
# sample. Heuristic: any short line (<=80 chars) that appears identically on
# a large fraction of "pages" (delimited by our own "[page N]" markers) is
# almost certainly boilerplate, not content, and gets dropped.
_HEADER_FOOTER_MAX_LINE_LEN = 80
_HEADER_FOOTER_MIN_PAGE_FRACTION = 0.3


def _strip_repeated_headers_footers(text: str) -> str:
    pages = text.split("\n\n[page ")
    if len(pages) < 4:  # not enough pages for repetition to be meaningful
        return text

    from collections import Counter
    line_counts: Counter = Counter()
    for page in pages:
        # Only look at the first/last couple of lines of each page, since
        # that's where headers/footers live -- not the whole page.
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        edge_lines = lines[:2] + lines[-2:]
        for ln in set(edge_lines):
            if len(ln) <= _HEADER_FOOTER_MAX_LINE_LEN:
                line_counts[ln] += 1

    threshold = max(3, int(len(pages) * _HEADER_FOOTER_MIN_PAGE_FRACTION))
    boilerplate = {ln for ln, count in line_counts.items() if count >= threshold}
    if not boilerplate:
        return text

    kept_lines = [ln for ln in text.splitlines() if ln.strip() not in boilerplate]
    return "\n".join(kept_lines)


# --------------------------------------------------------------------------
# Manifest (tracks what's already indexed per subject, so unchanged files
# are never re-embedded). Lives inside that subject's own vector db folder.
# --------------------------------------------------------------------------
def _manifest_path(subject: str) -> Path:
    return subject_chroma_dir(subject) / "manifest.json"


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Parse
# --------------------------------------------------------------------------
def _extract_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[page {i + 1}]\n{text}")
    return "\n\n".join(pages)


# --------------------------------------------------------------------------
# Clean
# --------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """Strip trailing whitespace per line, but -- unlike a naive
    'drop every blank line' pass -- collapse runs of blank lines down to a
    SINGLE blank line instead of removing them outright. Paragraph breaks
    (\\n\\n) are what _chunk_text uses to split the book into chunks; if we
    strip every blank line, a whole book can come through with no \\n\\n
    left at all and get treated as one giant unsplittable chunk."""
    text = _strip_repeated_headers_footers(text)
    lines = text.splitlines()
    cleaned: list[str] = []
    prev_blank = False
    for ln in lines:
        stripped = ln.strip()
        if stripped:
            cleaned.append(stripped)
            prev_blank = False
        elif not prev_blank:
            cleaned.append("")  # keep exactly one blank line as a paragraph break
            prev_blank = True
    return "\n".join(cleaned).strip()


# --------------------------------------------------------------------------
# Chunk (paragraph-aware, with 1-paragraph overlap for context continuity)
# --------------------------------------------------------------------------
def _split_long_block(block: str, max_len: int) -> list[str]:
    """Hard fallback for a 'paragraph' that turns out to be huge (e.g. a
    PDF whose extracted text has no real blank-line breaks at all, so the
    whole book comes through as a single block). Splits by max_len chars,
    preferring to break on whitespace near the boundary so we don't cut
    words in half."""
    if len(block) <= max_len:
        return [block]
    pieces = []
    start, n = 0, len(block)
    while start < n:
        end = min(start + max_len, n)
        if end < n:
            space = block.rfind(" ", start, end)
            if space > start:
                end = space
        piece = block[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end
    return pieces


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS) -> list[str]:
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        raw_paragraphs = [text] if text.strip() else []

    # Guarantee every unit we accumulate below is <= chunk_size -- this is
    # what actually prevents a whole book from becoming one embeddings call.
    paragraphs: list[str] = []
    for p in raw_paragraphs:
        paragraphs.extend(_split_long_block(p, chunk_size))

    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = current[-CHUNK_OVERLAP_PARAGRAPHS:] if CHUNK_OVERLAP_PARAGRAPHS else []
            current_len = sum(len(p) for p in current)
        current.append(para)
        current_len += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# --------------------------------------------------------------------------
# Per-subject ingestion
# --------------------------------------------------------------------------
def ingest_subject(subject: str, progress_cb=None) -> dict:
    """Ingest every PDF under Books/<subject>/ into that subject's own
    vector db, and (re)build the subject's topic checklist.

    progress_cb, if given, is called with a dict describing progress so far
    -- e.g. {"phase": "embedding", "file": "...", "file_index": 2,
    "total_files": 4, "chunk_index": 300, "total_chunks": 1536} -- so a
    caller (e.g. a background job) can report live status instead of just
    blocking silently. Embedding a large book is genuinely slow (one HTTP
    call per chunk to a local model), so this matters a lot in practice.
    """
    ensure_dirs()
    books_dir = subject_books_dir(subject)
    manifest_path = _manifest_path(subject)
    manifest = _load_manifest(manifest_path)
    collection = get_subject_collection(subject)

    def _report(**kwargs):
        if progress_cb:
            try:
                progress_cb(kwargs)
            except Exception:
                pass  # progress reporting must never break ingestion itself

    found_files = [p for p in books_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    log.info(f"[{subject}] scanning {books_dir}: found {len(found_files)} PDF(s)")
    found_keys = set()
    total_files = len(found_files)
    _report(phase="scanning", total_files=total_files)

    stats = {
        "subject": subject,
        "files_scanned": len(found_files),
        "files_new": 0,
        "files_updated": 0,
        "files_skipped": 0,
        "files_removed": 0,
        "chunks_added": 0,
        "files_no_extractable_text": [],
        "errors": [],
        "topics_total": 0,
    }

    # Text sample per book, used for topic extraction below -- collected
    # for every book that's new/updated, reused from the manifest for
    # unchanged books whose topics already extracted successfully, and
    # retried (topics only, no re-embedding) for unchanged books whose
    # topics extraction previously failed.
    per_book_topics: dict[str, list[str]] = {}

    for file_index, path in enumerate(found_files, start=1):
        rel_key = str(path.relative_to(books_dir))
        found_keys.add(rel_key)
        _report(phase="file_start", file=rel_key, file_index=file_index, total_files=total_files)

        try:
            file_hash = _file_hash(path)
        except OSError as e:
            log.error(f"[{subject}] {rel_key}: could not read file ({e})")
            stats["errors"].append(f"{rel_key}: could not read file ({e})")
            continue

        prior = manifest.get(rel_key)
        if prior and prior.get("hash") == file_hash:
            stats["files_skipped"] += 1
            cached_topics = prior.get("topics") or []
            if prior.get("topics_extracted", bool(cached_topics)):
                per_book_topics[rel_key] = cached_topics
                continue
            # Chunks are already indexed, but topic extraction previously
            # failed (or hasn't run yet, e.g. from before this field
            # existed) -- retry JUST the topic step. Cheap: re-reads and
            # re-cleans the PDF text, but does not re-chunk or re-embed
            # anything, so it's fast even for a huge book.
            log.info(f"[{subject}] {rel_key}: already indexed, retrying topic extraction only")
            try:
                retry_text = _clean_text(_extract_text(path))
            except Exception as e:
                log.error(f"[{subject}] {rel_key}: couldn't re-read file for topic retry ({e})")
                per_book_topics[rel_key] = cached_topics
                continue
            try:
                book_topics = extract_topics_from_book(rel_key, retry_text)
            except Exception as e:
                log.error(f"[{subject}] {rel_key}: topic retry crashed unexpectedly ({e})")
                book_topics = []
            manifest[rel_key]["topics"] = book_topics
            manifest[rel_key]["topics_extracted"] = bool(book_topics)
            per_book_topics[rel_key] = book_topics
            _save_manifest(manifest, manifest_path)
            continue

        if prior:
            old_ids = prior.get("chunk_ids", [])
            if old_ids:
                collection.delete(ids=old_ids)
            stats["files_updated"] += 1
        else:
            stats["files_new"] += 1

        try:
            extracted = _extract_text(path)
        except Exception as e:  # malformed PDF, encoding issue, etc.
            log.error(f"[{subject}] {rel_key}: failed to parse ({e})")
            stats["errors"].append(f"{rel_key}: failed to parse ({e})")
            continue

        if not extracted.strip():
            log.warning(f"[{subject}] {rel_key}: no extractable text (likely scanned/image-based)")
            stats["files_no_extractable_text"].append(rel_key)
            # topics_extracted=True here on purpose: there's no text to ever
            # extract topics from, so retrying would just fail the same way
            # forever -- this is a terminal state, not a transient failure.
            manifest[rel_key] = {"hash": file_hash, "chunk_ids": [], "topics": [], "topics_extracted": True}
            _save_manifest(manifest, manifest_path)
            continue

        raw_text = _clean_text(extracted)
        chunks = _chunk_text(raw_text)
        log.info(f"[{subject}] {rel_key}: extracted {len(raw_text)} chars -> {len(chunks)} chunk(s)")
        _report(
            phase="embedding", file=rel_key, file_index=file_index, total_files=total_files,
            chunk_index=0, total_chunks=len(chunks),
        )

        ids, embeddings, documents, metadatas = [], [], [], []
        batch_endpoint_broken = False  # set once if /api/embed itself isn't supported by this Ollama
        for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
            batch_embeddings: list[list[float] | None] = [None] * len(batch)

            if not batch_endpoint_broken:
                try:
                    batch_embeddings = embed_texts(EMBED_MODEL, batch, keep_alive=EMBED_KEEP_ALIVE_DURING_INGEST)
                except Exception as e:
                    log.warning(
                        f"[{subject}] {rel_key}: batch embedding failed ({e}), "
                        "falling back to one-at-a-time for the rest of this file"
                    )
                    batch_endpoint_broken = True
                    batch_embeddings = [None] * len(batch)

            if batch_endpoint_broken:
                for j, chunk in enumerate(batch):
                    try:
                        batch_embeddings[j] = embed_text(EMBED_MODEL, chunk, keep_alive=EMBED_KEEP_ALIVE_DURING_INGEST)
                    except Exception as e:
                        i = batch_start + j
                        log.error(f"[{subject}] {rel_key} chunk {i}: embedding failed ({e})")
                        stats["errors"].append(f"{rel_key} chunk {i}: embedding failed ({e})")

            for j, (chunk, embedding) in enumerate(zip(batch, batch_embeddings)):
                if embedding is None:
                    continue
                i = batch_start + j
                ids.append(f"{rel_key}::chunk::{i}")
                embeddings.append(embedding)
                documents.append(chunk)
                metadatas.append({"source": rel_key, "chunk_index": i})

            _report(
                phase="embedding", file=rel_key, file_index=file_index, total_files=total_files,
                chunk_index=min(batch_start + len(batch), len(chunks)), total_chunks=len(chunks),
            )

        if ids:
            collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        elif chunks:
            # We had real chunks to embed but every single one failed (e.g.
            # Ollama returned a 500) -- do NOT write a manifest entry for
            # this file. If we did, its hash would match on the next run
            # and it would be silently skipped forever, even though nothing
            # actually made it into the vector db. Leaving no entry means
            # it's retried as "new" next time.
            log.error(f"[{subject}] {rel_key}: all {len(chunks)} chunk(s) failed to embed, will retry next run")
            stats["errors"].append(f"{rel_key}: all chunks failed to embed, will retry next run")
            continue

        _report(phase="topics", file=rel_key, file_index=file_index, total_files=total_files)
        try:
            book_topics = extract_topics_from_book(rel_key, raw_text)
        except Exception as e:
            log.error(f"[{subject}] {rel_key}: topic extraction crashed unexpectedly ({e}), continuing without its topics")
            stats["errors"].append(f"{rel_key}: topic extraction failed ({e})")
            book_topics = []
        per_book_topics[rel_key] = book_topics

        manifest[rel_key] = {
            "hash": file_hash,
            "chunk_ids": ids,
            "topics": book_topics,
            "topics_extracted": bool(book_topics),
        }
        stats["chunks_added"] += len(ids)
        log.info(f"[{subject}] {rel_key}: indexed {len(ids)} chunk(s), {len(book_topics)} candidate topic(s)")

        # Save after EVERY file, not just once at the very end -- ingesting
        # a whole subject can take hours, and a single book can take 30+
        # minutes on its own. If the process is interrupted, already-
        # finished books should stay indexed rather than being silently
        # lost and re-embedded from scratch next time.
        _save_manifest(manifest, manifest_path)
        _report(phase="file_done", file=rel_key, file_index=file_index, total_files=total_files)

    # Remove entries for files that no longer exist in this subject's folder
    removed_keys = set(manifest.keys()) - found_keys
    for key in removed_keys:
        old_ids = manifest[key].get("chunk_ids", [])
        if old_ids:
            collection.delete(ids=old_ids)
        del manifest[key]
        stats["files_removed"] += 1
        log.info(f"[{subject}] {key}: removed (no longer in folder)")

    _save_manifest(manifest, manifest_path)

    _report(phase="topics_merge")
    try:
        merged_topics = merge_subject_topics(subject, per_book_topics)
    except Exception as e:
        log.error(f"[{subject}] topic merge crashed unexpectedly ({e}) -- books are still indexed, just without an updated checklist this run")
        stats["errors"].append(f"topic checklist merge failed: {e}")
        merged_topics = []
    stats["topics_total"] = len(merged_topics)

    log.info(f"[{subject}] ingestion complete: {stats}")
    _report(phase="done", stats=stats)
    return stats


def ingest_all_subjects(progress_cb=None) -> dict:
    results = {}
    for idx, subject in enumerate(SUBJECTS, start=1):
        if progress_cb:
            try:
                progress_cb({"phase": "subject_start", "subject": subject, "subject_index": idx, "total_subjects": len(SUBJECTS)})
            except Exception:
                pass

        def _subject_progress_cb(update, _subject=subject, _idx=idx):
            if progress_cb:
                try:
                    progress_cb({**update, "subject": _subject, "subject_index": _idx, "total_subjects": len(SUBJECTS)})
                except Exception:
                    pass

        results[subject] = ingest_subject(subject, progress_cb=_subject_progress_cb)
    return results


def subject_status(subject: str) -> dict:
    manifest = _load_manifest(_manifest_path(subject))
    files = [
        {"source": k, "chunks": len(v.get("chunk_ids", []))}
        for k, v in manifest.items()
    ]
    files.sort(key=lambda f: f["source"])
    return {
        "subject": subject,
        "files": files,
        "total_files": len(files),
        "total_chunks": sum(f["chunks"] for f in files),
    }


def all_subjects_status() -> list[dict]:
    return [subject_status(subject) for subject in SUBJECTS]
