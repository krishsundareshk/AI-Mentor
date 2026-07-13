# imports
import json
import re

import json5

from config import QWEN_MODEL, EXPLAINER_MODEL, RETRIEVAL_TOP_K, OLLAMA_NUM_GPU
from ollama_client import call_model
from retrieval import retrieve
from settings import get_settings
from logger import get_logger
import prompts

log = get_logger("orchestrator")

# --------------------------------------------------------------------------
# Context window sizing
# --------------------------------------------------------------------------
# Ollama defaults to a 4096-token context unless told otherwise, and silently
# rejects (400 exceed_context_size_error) once system prompt + retrieved
# context + question exceed it. Every call below now passes num_ctx
# explicitly instead of relying on the model's default. 8192 comfortably
# covers a full system prompt + RETRIEVAL_TOP_K chunks + question; bump this
# if you raise RETRIEVAL_TOP_K or chunk size later.
DEFAULT_NUM_CTX = 8192

# The three "teaching" calls (teach_code, explain_concept, teach_leetcode) ask
# DeepSeek-R1 for a full, in-depth JSON card -- 8/9 fully-written steps plus
# line-by-line explanations. That's a LOT of output tokens, on top of
# whatever the model spends inside <think>...</think> before it even starts
# the JSON. num_ctx caps prompt+response tokens TOGETHER, so at 8192 it's
# very easy for a long answer to get cut off mid-JSON -- which then fails to
# parse and silently falls back to the mostly-empty "unstructured" card
# (see the except blocks below). Give these three calls real headroom.
TEACHING_NUM_CTX = 16384

# --------------------------------------------------------------------------
# Code-vs-concept routing
# --------------------------------------------------------------------------
# Per your instruction: call Qwen only when code is actually needed. Pure
# concept questions (which will be most of your DSA/SQL/ML/DL/backend/agentic
# AI interview prep) go to DeepSeek alone -- faster, and no code is invented
# where none was asked for.
_CODE_SIGNAL_PATTERNS = [
    r"\bwrite\b", r"\bimplement\b", r"\brefactor\b", r"\bdebug\b", r"\bfix this\b", r"\bfix my\b",
    r"```", r"\boptimi[sz]e this\b", r"\bcode review\b",
    r"\bgive me (the |a )?code\b", r"\bshow me (the |a )?code\b",
    r"\bsql query\b", r"\bwrite a query\b", r"\bcode up\b",
]
_CODE_SIGNAL_RE = re.compile("|".join(_CODE_SIGNAL_PATTERNS), re.IGNORECASE)


def looks_like_code_request(question: str) -> bool:
    """Cheap heuristic, no extra model call -- keeps concept questions fast."""
    return bool(_CODE_SIGNAL_RE.search(question))


def _explanation_style() -> str:
    return get_settings().get("explanation_style", "")


_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_code(qwen_response: str) -> str:
    match = _CODE_BLOCK_RE.search(qwen_response)
    if match:
        return match.group(1).strip()
    # Fallback: no fenced block found, return the raw response
    return qwen_response.strip()


def _strip_think(text: str) -> str:
    """deepseek-r1 emits its own reasoning wrapped in <think>...</think>.
    Strip it before we try to parse JSON out of the response."""
    return _THINK_TAG_RE.sub("", text).strip()


def _attempt_json_repair(text: str) -> dict | None:
    """Best-effort recovery for a JSON object that got cut off mid-generation
    (e.g. the model hit num_ctx before finishing the last field/string). This
    is what actually prevents "many fields empty" -- without it, a response
    that's 95% complete but missing a final closing brace was being thrown
    away entirely in favor of the single-field fallback card. Tries closing
    the structure as-is first, then progressively trims the last (possibly
    unterminated) key/value pair and tries again."""
    if not text:
        return None

    candidates = [text]
    last_comma = text.rfind(",")
    if last_comma != -1:
        candidates.append(text[:last_comma])

    for raw in candidates:
        candidate = raw.rstrip()
        if not candidate:
            continue
        # An odd number of unescaped double quotes means the last string
        # literal was never closed -- close it before balancing brackets.
        if candidate.count('"') % 2 == 1:
            candidate += '"'
        candidate = candidate.rstrip().rstrip(",")
        opens_curly = candidate.count("{") - candidate.count("}")
        opens_square = candidate.count("[") - candidate.count("]")
        candidate += "]" * max(opens_square, 0)
        candidate += "}" * max(opens_curly, 0)
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            try:
                return json5.loads(candidate)
            except Exception:
                continue
    return None


def _extract_json(text: str) -> dict:
    text = _strip_think(text)
    fence_match = re.search(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        text = brace_match.group(0)
    else:
        # No closing brace found at all -- the response was almost certainly
        # cut off mid-generation. Work from the first '{' onward so the
        # repair step below has a real (if incomplete) object to close.
        start = text.find("{")
        if start != -1:
            text = text[start:]

    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    try:
        return json5.loads(text)
    except Exception:
        pass

    repaired = _attempt_json_repair(text)
    if repaired is not None:
        log.warning("_extract_json: response was truncated/malformed, recovered via repair")
        return repaired

    raise ValueError("Could not parse or repair JSON from model response")


def _clean_code_field(code: str) -> str:
    stripped = code.strip()
    lines = stripped.split("\n", 1)
    if len(lines) == 2 and lines[0].strip().lower() in {"python", "python3", "py"}:
        return lines[1]
    return stripped


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# --------------------------------------------------------------------------
# Code generation / code teaching -- NEVER grounded in the subject book
# library. `context`, when passed, is the student's own project code
# (workspace), not the knowledge base.
# --------------------------------------------------------------------------
def generate_code(question: str, context: str = "") -> str:
    """Call Qwen to produce code for the student's question. `context`, if
    given, is retrieved from the WORKSPACE (the student's own project code)
    -- never from a subject's book library."""
    user_content = question
    if context:
        user_content = (
            f"Relevant code from the student's own project:\n{context}\n\n"
            f"Student's question:\n{question}"
        )
    response = call_model(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": prompts.QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        keep_alive=0,  # unload Qwen right away -- DeepSeek is called next and needs the VRAM
        num_ctx=DEFAULT_NUM_CTX,
        num_gpu=OLLAMA_NUM_GPU,
    )
    return _extract_code(response)


def teach_code(question: str, code: str, context: str = "") -> dict:
    """Call Qwen 3 to produce the code-teaching Markdown response. `context`, if given,
    is workspace (project code) context -- never the subject book library."""
    context_block = (
        f"Relevant code from the student's own project:\n{context}\n\n"
        if context
        else ""
    )
    user_content = (
        f"{context_block}"
        f"Student's question:\n{question}\n\n"
        f"Code produced by Qwen:\n```python\n{code}\n```\n\n"
        "Explain this code in full depth using beautiful formatted Markdown."
    )
    raw = call_model(
        model=EXPLAINER_MODEL,
        messages=[
            {"role": "system", "content": prompts.CODE_TEACHING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        num_ctx=TEACHING_NUM_CTX,
        num_gpu=OLLAMA_NUM_GPU,
    )
    return {"text": raw}


# --------------------------------------------------------------------------
# Subject Explainer -- the ONLY path that retrieves from a subject's book
# library. Uses the 8-step framework, with a per-subject teaching style.
# --------------------------------------------------------------------------
def explain_concept(question: str, context: str = "", subject: str | None = None) -> dict:
    """Deep explanation of a concept using Qwen 3 with Markdown."""
    context_block = (
        f"Relevant documentation retrieved from the student's own library:\n{context}\n\n"
        if context
        else ""
    )
    user_content = (
        f"{context_block}"
        f"Student's question:\n{question}\n\n"
        "Explain this concept in full depth using beautiful formatted Markdown."
    )
    system_prompt = prompts.get_subject_explainer_prompt(subject)
    raw = call_model(
        model=EXPLAINER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        num_ctx=TEACHING_NUM_CTX,
        num_gpu=OLLAMA_NUM_GPU,
    )
    return {"text": raw}


def run_teaching_turn(question: str, include_workspace: bool = False, mode_hint: str = "auto") -> dict:
    """Full pipeline for Code Help / Project Mentor. `include_workspace`
    controls whether the student's own project code is retrieved as
    context -- this never touches the subject book library, only the
    workspace collection.

    mode_hint: "auto" (heuristic decides), "code" (force Qwen+DeepSeek),
    or "concept" (force DeepSeek alone).
    """
    workspace_chunks = (
        retrieve(question, top_k=RETRIEVAL_TOP_K, collection_name="workspace")
        if include_workspace
        else []
    )
    context = _format_context(workspace_chunks)

    if mode_hint == "code":
        use_code = True
    elif mode_hint == "concept":
        use_code = False
    else:
        use_code = looks_like_code_request(question)

    log.info(f"run_teaching_turn: mode_hint={mode_hint} -> use_code={use_code} | question={question[:80]!r}")

    if use_code:
        code = generate_code(question, context)
        card = teach_code(question, code, context)
    else:
        # Generic concept question, not tied to a subject -- the
        # non-subject-scoped explainer prompt. No book-library retrieval
        # happens here by design; only workspace context (if enabled),
        # same as the code path above.
        card = explain_concept(question, context, subject=None)

    card["detected_mode"] = "code" if use_code else "concept"

    if not card.get("sources") and workspace_chunks:
        card["sources"] = [
            {"source": c["source"], "note": "retrieved as relevant context"} for c in workspace_chunks
        ]
    return card


def run_subject_turn(subject: str, question: str, mode_hint: str = "auto") -> dict:
    """Full pipeline for a subject question (Python, DSA, OS, ...).

    Code path: Qwen generates, DeepSeek teaches the code -- NEITHER call is
    grounded in the subject's book library, per your instruction that code
    help shouldn't reference the knowledge base.

    Concept path: DeepSeek alone, using this subject's 8-step explainer
    prompt, grounded in that subject's own vector db -- this is the only
    place subject books are ever retrieved.
    """
    if mode_hint == "code":
        use_code = True
    elif mode_hint == "concept":
        use_code = False
    else:
        use_code = looks_like_code_request(question)

    log.info(f"run_subject_turn[{subject}]: mode_hint={mode_hint} -> use_code={use_code} | question={question[:80]!r}")

    chunks = []
    if use_code:
        code = generate_code(question)          # no KB context
        card = teach_code(question, code)       # no KB context
    else:
        chunks = retrieve(question, top_k=RETRIEVAL_TOP_K, subject=subject)
        context = _format_context(chunks)
        card = explain_concept(question, context, subject=subject)

    card["detected_mode"] = "code" if use_code else "concept"
    card["subject"] = subject

    if not card.get("sources") and chunks:
        card["sources"] = [
            {"source": c["source"], "note": "retrieved as relevant context"} for c in chunks
        ]
    return card


# --------------------------------------------------------------------------
# LeetCode mode -- code help, not subject-explainer, so no KB retrieval
# (same rule as run_subject_turn's code path). Qwen generates both
# solutions; DeepSeek teaches both using the 9-step framework.
# --------------------------------------------------------------------------
def generate_leetcode_solutions(problem: str) -> tuple[str, str]:
    """Returns (brute_force_code, optimized_code) -- two separate Qwen calls
    so each solution is cleanly extracted on its own, rather than parsing
    two code blocks out of one response."""
    brute_response = call_model(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": prompts.LEETCODE_BRUTEFORCE_QWEN_PROMPT},
            {"role": "user", "content": problem},
        ],
        temperature=0.2,
        num_ctx=DEFAULT_NUM_CTX,
        num_gpu=OLLAMA_NUM_GPU,
        # NOT keep_alive=0 here -- the optimized-solution call below reuses
        # the same Qwen model right away, so keep it warm in between.
    )
    brute_code = _extract_code(brute_response)

    optimized_response = call_model(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": prompts.LEETCODE_OPTIMIZED_QWEN_PROMPT},
            {"role": "user", "content": problem},
        ],
        temperature=0.2,
        keep_alive=0,  # last Qwen call before DeepSeek -- free VRAM now
        num_ctx=DEFAULT_NUM_CTX,
        num_gpu=OLLAMA_NUM_GPU,
    )
    optimized_code = _extract_code(optimized_response)

    return brute_code, optimized_code


def teach_leetcode(problem: str, brute_code: str, optimized_code: str) -> dict:
    user_content = (
        f"Problem:\n{problem}\n\n"
        f"Brute force solution:\n```python\n{brute_code}\n```\n\n"
        f"Optimized solution:\n```python\n{optimized_code}\n```\n\n"
        "Produce the full LeetCode explanation covering both solutions using formatted Markdown."
    )
    raw = call_model(
        model=EXPLAINER_MODEL,
        messages=[
            {"role": "system", "content": prompts.LEETCODE_EXPLAINER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        num_ctx=TEACHING_NUM_CTX,
        num_gpu=OLLAMA_NUM_GPU,
    )
    return {"text": raw}


def run_leetcode_turn(problem: str) -> dict:
    """Full pipeline for a LeetCode-style problem: brute force + optimized
    solutions (Qwen, no KB) -> teach both (DeepSeek, 9-step framework, no KB)."""
    brute_code, optimized_code = generate_leetcode_solutions(problem)
    card = teach_leetcode(problem, brute_code, optimized_code)
    return card


def explain_topic_concepts(topic: str, context: str, subject: str) -> dict:
    """Specialized explanation turn for a specific topic, explaining concrete
    concepts present in the database chunks."""
    context_block = (
        f"Database chunks retrieved for the topic '{topic}':\n{context}\n\n"
        if context
        else ""
    )
    user_content = (
        f"{context_block}"
        f"Explain all the technical concepts and techniques described in the database chunks above. "
        f"Do not write a meta-explanation of the phrase '{topic}'. Instead, teach the actual "
        f"concepts, tools, and steps that are described in the database chunks."
    )
    system_prompt = prompts.get_subject_explainer_prompt(subject)
    raw = call_model(
        model=EXPLAINER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        num_ctx=TEACHING_NUM_CTX,
        num_gpu=OLLAMA_NUM_GPU,
    )
    return {"text": raw}


def explain_topic_followup(topic: str, question: str, chat_history: list[dict], context: str, subject: str) -> dict:
    """Run a follow-up turn in topic-specific chat history with grounded database context."""
    messages = []
    
    # 1. Add system prompt
    system_prompt = prompts.get_subject_explainer_prompt(subject)
    system_prompt += (
        "\n\nThis is a follow-up discussion. Answer the student's question directly, "
        "referencing previous messages in the conversation and the provided database context "
        "where appropriate. Maintain the same detailed and elaborative markdown teaching style."
    )
    messages.append({"role": "system", "content": system_prompt})
    
    # 2. Add database context
    if context:
        context_msg = f"Below is the relevant reference text from the book database to ground this discussion:\n{context}"
        messages.append({"role": "system", "content": context_msg})
    
    # 3. Add previous chat history
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    # 4. Add the new followup question
    messages.append({"role": "user", "content": question})
    
    raw = call_model(
        model=EXPLAINER_MODEL,
        messages=messages,
        temperature=0.3,
        num_ctx=TEACHING_NUM_CTX,
        num_gpu=OLLAMA_NUM_GPU,
    )
    return {"text": raw}
