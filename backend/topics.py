"""Per-subject topic checklists.

After a subject's books are ingested, we ask DeepSeek to read a sample of
each book and list the topics it covers, then merge every book's list into
one clean, deduplicated master checklist for the subject (e.g. all of your
5 DSA books collapse into a single "Binary Search Trees" entry instead of
5 near-duplicates). The checklist is persisted as JSON so ticking a topic
off survives re-ingestion -- re-running ingestion only adds newly-discovered
topics, it never resets your progress.
"""
import json
from pathlib import Path

import math

from config import subject_topics_path, ensure_dirs
from ollama_client import call_model, embed_texts, OllamaError
from config import EXPLAINER_MODEL, QWEN_MODEL, EMBED_MODEL
from logger import get_logger

log = get_logger("topics")

# How much of a book's cleaned text to actually show the model when asking
# "what topics does this book cover" -- covers title/TOC/intro plus a solid
# sample of body content without blowing the context window per book.
# Total characters shown to the model for topic extraction, per book.
TOPIC_SAMPLE_CHARS = 6000
# How many evenly-spaced points across the book to pull a slice from. A
# single head-of-book sample (the old behavior) only ever sees the title
# page/TOC/intro -- topics from the middle and back half of a 300+ page book
# were never visible to the model. Spreading the same total budget across
# several points gives real coverage of later chapters instead.
TOPIC_SAMPLE_POINTS = 4
TOPIC_EXTRACTION_NUM_CTX = 8192  # override Ollama's default 4096 -- our prompt + sample can exceed it
MAX_CANDIDATE_TOPICS_FOR_MERGE = 400  # safety cap before the merge call

# Embedding-based near-duplicate clustering, run BEFORE any merge LLM call.
# Two candidate topic strings whose embeddings have cosine similarity >=
# this threshold are treated as the same topic (e.g. "Big-O Notation" vs
# "Time Complexity (Big O)"). This is what actually keeps the merge step
# fast: for a subject with 5+ books you can easily end up with 300-400 raw
# candidates, and previously ALL of them were dumped into a single prompt
# for deepseek-r1 (a reasoning model) to dedupe by "thinking" -- that one
# call was the entire bottleneck in ingestion. Embedding similarity does
# the same dedupe job in milliseconds using the embedding model we already
# call constantly elsewhere, so the LLM only ever sees a handful of
# already-deduped cluster representatives.
TOPIC_DEDUPE_SIMILARITY_THRESHOLD = 0.86
# If embedding clustering alone gets us down to this many topics or fewer,
# skip the LLM cleanup call entirely -- there's nothing left worth an LLM's
# time (just title-case the cluster representatives).
SKIP_LLM_CLEANUP_BELOW = 15

_PER_BOOK_SYSTEM_PROMPT = """You extract a table-of-contents-style list of \
topics from a technical book excerpt. Return ONLY a JSON array of short \
topic name strings (e.g. ["Binary Search Trees", "AVL Rotations"]) -- no \
markdown fences, no commentary, no nested objects. Use concise, standard \
terminology a student would recognize (title-case, 1-5 words each). List \
every distinct topic you can identify from the excerpt, typically 10-40 \
items for a full technical book. If the excerpt is too short or unclear to \
identify real topics, return an empty array []."""

_MERGE_SYSTEM_PROMPT = """You are cleaning up a topic checklist for a study \
app. You will receive a list of candidate topic names collected from \
several different books on the same subject -- it will contain many exact \
or near duplicates (different books naming the same idea slightly \
differently, e.g. "Big-O Notation" and "Time Complexity (Big O)"). \
Merge these into ONE clean, deduplicated master list of unique topics, \
using clear, standard, student-friendly names. Preserve genuinely distinct \
topics as separate entries. Return ONLY a JSON array of strings, no \
markdown fences, no commentary."""


def _load(subject: str) -> dict:
    path = subject_topics_path(subject)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"topics": {}}  # name -> {"done": bool}


def _save(subject: str, data: dict) -> None:
    ensure_dirs()
    path = subject_topics_path(subject)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse_json_array(raw: str) -> list[str]:
    text = raw.strip()
    # Strip <think> traces / fences defensively, same spirit as orchestrator.
    if "<think>" in text.lower():
        end = text.lower().rfind("</think>")
        if end != -1:
            text = text[end + len("</think>"):].strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if str(t).strip()]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def _spread_sample(text: str, total_chars: int, num_points: int) -> str:
    """Pull `num_points` evenly-spaced slices from across `text` (each
    total_chars/num_points long) instead of just the head. The first slice
    is always the very start of the book (title/TOC/intro still matter),
    and the rest are spread through the middle and end so later chapters
    actually get a chance to surface topics."""
    n = len(text)
    if n <= total_chars or num_points <= 1:
        return text[:total_chars]

    per_slice = total_chars // num_points
    slices = []
    for i in range(num_points):
        start = (n * i) // num_points
        end = min(start + per_slice, n)
        slices.append(text[start:end])
    return "\n\n[...]\n\n".join(slices)


def extract_topics_from_book(book_name: str, cleaned_text: str) -> list[str]:
    """Ask DeepSeek what topics a single book covers, from text sampled
    across the whole book (not just the first few pages)."""
    sample = _spread_sample(cleaned_text, TOPIC_SAMPLE_CHARS, TOPIC_SAMPLE_POINTS)
    if not sample.strip():
        return []
    try:
        raw = call_model(
            model=EXPLAINER_MODEL,
            messages=[
                {"role": "system", "content": _PER_BOOK_SYSTEM_PROMPT},
                {"role": "user", "content": f"Book: {book_name}\n\nExcerpt:\n{sample}"},
            ],
            temperature=0.1,
            num_ctx=TOPIC_EXTRACTION_NUM_CTX,
        )
    except OllamaError as e:
        log.error(f"topic extraction failed for {book_name}: {e}")
        return []
    return _parse_json_array(raw)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _embedding_dedupe(candidates: list[str], threshold: float = TOPIC_DEDUPE_SIMILARITY_THRESHOLD) -> list[str]:
    """Collapse near-duplicate topic names using embedding similarity instead
    of an LLM call. Greedy single-pass clustering: walk the candidates in
    order, and for each one either fold it into the first existing cluster
    it's similar enough to, or start a new cluster with it as the
    representative. O(n^2) cosine comparisons, but n is at most a few
    hundred short strings and the comparisons themselves are cheap -- the
    only network cost is ONE batched /api/embed call for all candidates,
    which is milliseconds compared to an LLM generation call."""
    if not candidates:
        return []
    try:
        vectors = embed_texts(EMBED_MODEL, candidates)
    except OllamaError as e:
        log.warning(f"embedding dedupe failed ({e}), skipping to raw candidate list")
        return candidates

    representatives: list[str] = []
    rep_vectors: list[list[float]] = []
    for name, vec in zip(candidates, vectors):
        matched = False
        for i, rep_vec in enumerate(rep_vectors):
            if _cosine(vec, rep_vec) >= threshold:
                matched = True
                break
        if not matched:
            representatives.append(name)
            rep_vectors.append(vec)
    return representatives


def _merge_candidates(subject: str, candidates: list[str]) -> list[str]:
    if not candidates:
        return []
    # Cheap pre-dedupe (case/whitespace-insensitive) before anything else,
    # to keep later steps a reasonable size.
    seen = {}
    for c in candidates:
        key = " ".join(c.lower().split())
        if key not in seen:
            seen[key] = c
    unique_candidates = list(seen.values())[:MAX_CANDIDATE_TOPICS_FOR_MERGE]

    # Embedding-based near-duplicate clustering FIRST. This is the step that
    # actually removes near-duplicates like "Big-O Notation" vs "Time
    # Complexity (Big O)" -- and it does it in milliseconds via one batched
    # embedding call, instead of asking a reasoning model to eyeball 400
    # strings in a single prompt (which was the real bottleneck: deepseek-r1
    # would burn minutes of "thinking" tokens on that comparison task).
    clustered = _embedding_dedupe(unique_candidates)

    if len(clustered) <= SKIP_LLM_CLEANUP_BELOW:
        # Not enough left to bother an LLM with -- just tidy casing.
        return list({" ".join(w.capitalize() for w in c.split()) for c in clustered})

    try:
        raw = call_model(
            # Non-reasoning model on purpose: naming/title-case cleanup on an
            # already-deduped list is straightforward text processing, not a
            # task that benefits from chain-of-thought. This alone cuts the
            # generation time from minutes to seconds versus deepseek-r1.
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": _MERGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Subject: {subject}\n\nCandidate topics:\n" + json.dumps(clustered)},
            ],
            temperature=0.1,
            keep_alive=0,
            num_ctx=TOPIC_EXTRACTION_NUM_CTX,
            timeout=300,  # the input list is now small, so this should be fast
        )
    except OllamaError as e:
        log.error(f"topic merge cleanup failed for subject {subject}: {e}")
        return clustered  # fall back to the embedding-deduped list, still good

    merged = _parse_json_array(raw)
    return merged if merged else clustered


def sort_topics_pedagogically(subject: str, topic_names: list[str]) -> list[str]:
    """Ask Qwen to sort the checklist topics logically from beginner to advanced."""
    if not topic_names:
        return []
    
    prompt = (
        f"You are a pedagogical curriculum assistant. You will receive a list of topics for the subject '{subject}'. "
        f"Sort these topics in a logical learning sequence from beginner (fundamental concepts, setup, basics) "
        f"to advanced (optimization, internals, complex patterns, edge cases).\n\n"
        f"Return ONLY a JSON array of the sorted topic name strings. Do not include markdown fences, "
        f"commentary, or any other text. Keep the topic names exactly as provided in the input list, just change their order."
    )
    try:
        raw = call_model(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(topic_names)},
            ],
            temperature=0.1,
            keep_alive=0,
            num_ctx=TOPIC_EXTRACTION_NUM_CTX,
        )
        sorted_list = _parse_json_array(raw)
        if sorted_list and len(sorted_list) == len(topic_names):
            if set(sorted_list) == set(topic_names):
                log.info(f"Pedagogical sorting completed successfully for {subject}")
                return sorted_list
    except Exception as e:
        log.error(f"Pedagogical sorting failed for {subject}: {e}")
        
    return sorted(topic_names)  # fallback to alphabetical if sorting fails


def merge_subject_topics(subject: str, per_book_topics: dict[str, list[str]]) -> list[str]:
    """Combine every book's candidate topics for a subject into one clean
    master list, then merge with the existing checklist -- new topics are
    added, existing topics (and their done/undone state) are preserved,
    and nothing already ticked off gets reset."""
    all_candidates: list[str] = []
    for topics in per_book_topics.values():
        all_candidates.extend(topics)

    if not all_candidates:
        data = _load(subject)
        return list(data.get("topics", {}).keys())

    clean_master_list = _merge_candidates(subject, all_candidates)

    data = _load(subject)
    existing = data.setdefault("topics", {})
    for name in clean_master_list:
        if name not in existing:
            existing[name] = {"done": False}
            
    # Invalidate pedagogical sort flag to trigger re-sorting on next load
    data["sorted_pedagogical"] = False
    _save(subject, data)
    return list(existing.keys())


def get_checklist(subject: str) -> list[dict]:
    data = _load(subject)
    topics = data.get("topics", {})
    
    # One-time pedagogical sort if not already done
    if not data.get("sorted_pedagogical") and topics:
        log.info(f"Performing one-time pedagogical sort for subject {subject}")
        ordered_names = sort_topics_pedagogically(subject, list(topics.keys()))
        
        new_topics = {}
        for name in ordered_names:
            if name in topics:
                new_topics[name] = topics[name]
        
        data["topics"] = new_topics
        data["sorted_pedagogical"] = True
        _save(subject, data)
        topics = new_topics
        
    return [
        {"name": name, "done": bool(info.get("done", False))}
        for name, info in topics.items()
    ]


def set_topic_done(subject: str, topic_name: str, done: bool) -> list[dict]:
    data = _load(subject)
    topics = data.setdefault("topics", {})
    if topic_name not in topics:
        raise KeyError(f"Unknown topic '{topic_name}' for subject '{subject}'")
    topics[topic_name]["done"] = bool(done)
    _save(subject, data)
    return get_checklist(subject)


def progress_summary(subject: str) -> dict:
    checklist = get_checklist(subject)
    done = sum(1 for t in checklist if t["done"])
    return {"total": len(checklist), "done": done, "remaining": len(checklist) - done}
