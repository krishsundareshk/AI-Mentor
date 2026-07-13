import traceback
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import memory_db
from orchestrator import (
    run_teaching_turn,
    generate_code,
    teach_code,
    explain_concept,
    explain_topic_concepts,
    explain_topic_followup,
    run_leetcode_turn,
    generate_leetcode_solutions,
    teach_leetcode,
    run_subject_turn,
    looks_like_code_request,
    _format_context,
)
from retrieval import retrieve
from config import RETRIEVAL_TOP_K, SUBJECTS, subject_from_slug, subject_slug
from ollama_client import OllamaError
from pdf_export import export_session_pdf
from ingest import ingest_subject, ingest_all_subjects, subject_status, all_subjects_status
import topics as topics_module
import workspace
import settings as settings_module
from logger import get_logger

log = get_logger("main")

app = FastAPI(title="Study Buddy")

# Mount web dashboard static files from the separate frontend folder
import os
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")


@app.on_event("startup")
def _startup():
    memory_db.init_db()


# --------------------------------------------------------------------------
# Global safety net: NEVER let an unhandled exception silently vanish.
# Every failure is logged with a full traceback to K:\AI-Mentor\data\app.log
# AND returned to the frontend as a real error message, instead of a bare
# 500 with no explanation.
# --------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    log.error(f"Unhandled exception on {request.method} {request.url.path}:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"{type(exc).__name__}: {exc}. Full traceback written to "
            "K:\\AI-Mentor\\data\\app.log"
        },
    )


class NewSessionRequest(BaseModel):
    title: str = "Untitled session"


class ChatRequest(BaseModel):
    session_id: int
    question: str
    mode_hint: str = "auto"  # "auto" | "code" | "concept"


class CodegenRequest(BaseModel):
    session_id: int
    question: str
    include_workspace: bool = False


class TeachRequest(BaseModel):
    session_id: int
    question: str
    code: str
    include_workspace: bool = False
    mode: str = "code"


class ConceptRequest(BaseModel):
    session_id: int
    question: str
    include_workspace: bool = False
    mode: str = "concept"


class LeetCodeRequest(BaseModel):
    session_id: int
    problem: str


class LeetCodeTeachRequest(BaseModel):
    session_id: int
    problem: str
    brute_code: str
    optimized_code: str


class SubjectChatRequest(BaseModel):
    session_id: int
    question: str
    mode_hint: str = "auto"  # "auto" | "code" | "concept"


class TopicToggleRequest(BaseModel):
    topic: str
    done: bool


class WorkspaceSetRequest(BaseModel):
    path: str


class SettingsRequest(BaseModel):
    explanation_style: str | None = None


class TopicExplainRequest(BaseModel):
    session_id: int
    topic: str


class TopicFollowupRequest(BaseModel):
    session_id: int
    topic: str
    question: str


def _require_session(session_id: int):
    session = memory_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _combined_context(question: str, include_workspace: bool):
    chunks = (
        retrieve(question, top_k=RETRIEVAL_TOP_K, collection_name="workspace")
        if include_workspace
        else []
    )
    return _format_context(chunks), chunks


def _require_subject(subject_slug_value: str) -> str:
    subject = subject_from_slug(subject_slug_value)
    if not subject:
        raise HTTPException(status_code=404, detail=f"Unknown subject slug '{subject_slug_value}'")
    return subject


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------
@app.post("/session")
def create_session(req: NewSessionRequest):
    session_id = memory_db.create_session(req.title)
    return {"session_id": session_id, "title": req.title}


@app.get("/sessions")
def list_sessions():
    return memory_db.list_sessions()


@app.get("/session/{session_id}/messages")
def get_messages(session_id: int, mode: str | None = None):
    _require_session(session_id)
    return memory_db.get_messages(session_id, mode=mode)


# --------------------------------------------------------------------------
# Settings (explanation style)
# --------------------------------------------------------------------------
@app.get("/settings")
def get_settings_endpoint():
    return settings_module.get_settings()


@app.post("/settings")
def update_settings_endpoint(req: SettingsRequest):
    return settings_module.update_settings(explanation_style=req.explanation_style)


# --------------------------------------------------------------------------
# Code Help / Project Mentor -- single-call convenience endpoints
# (kept for simplicity; the frontend uses the progressive /codegen + /teach
# or /concept endpoints below so you can SEE each sequential step complete)
# --------------------------------------------------------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    _require_session(req.session_id)
    try:
        card = run_teaching_turn(req.question, include_workspace=False, mode_hint=req.mode_hint)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))
    memory_db.add_message(req.session_id, req.question, card, mode="code")
    return card


@app.post("/project/chat")
def project_chat(req: ChatRequest):
    _require_session(req.session_id)
    try:
        card = run_teaching_turn(req.question, include_workspace=True, mode_hint=req.mode_hint)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))
    memory_db.add_message(req.session_id, req.question, card, mode="project")
    return card


# --------------------------------------------------------------------------
# Progressive endpoints: each model call is its own round trip, so the UI
# can show Qwen's code the moment it's ready, then DeepSeek's lesson --
# instead of one long black-box wait with no visible progress.
# --------------------------------------------------------------------------
@app.post("/codegen")
def codegen(req: CodegenRequest):
    _require_session(req.session_id)
    context, _ = _combined_context(req.question, req.include_workspace)
    try:
        code = generate_code(req.question, context)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"code": code}


@app.post("/teach")
def teach(req: TeachRequest):
    _require_session(req.session_id)
    context, chunks = _combined_context(req.question, req.include_workspace)
    try:
        card = teach_code(req.question, req.code, context)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))
    if not card.get("sources") and chunks:
        card["sources"] = [{"source": c["source"], "note": "retrieved as relevant context"} for c in chunks]
    card["detected_mode"] = "code"
    memory_db.add_message(req.session_id, req.question, card, mode=req.mode)
    return card


@app.post("/concept")
def concept(req: ConceptRequest):
    _require_session(req.session_id)
    context, chunks = _combined_context(req.question, req.include_workspace)
    try:
        card = explain_concept(req.question, context)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))
    if not card.get("sources") and chunks:
        card["sources"] = [{"source": c["source"], "note": "retrieved as relevant context"} for c in chunks]
    card["detected_mode"] = "concept"
    memory_db.add_message(req.session_id, req.question, card, mode=req.mode)
    return card


@app.post("/route")
def route(req: CodegenRequest):
    """Tells the frontend whether this question needs Qwen or DeepSeek-alone,
    so it knows which progressive path to take."""
    return {"use_code": looks_like_code_request(req.question)}


# --------------------------------------------------------------------------
# LeetCode -- both single-call and progressive
# --------------------------------------------------------------------------
@app.post("/leetcode")
def leetcode(req: LeetCodeRequest):
    _require_session(req.session_id)
    try:
        card = run_leetcode_turn(req.problem)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))
    memory_db.add_message(req.session_id, req.problem, card, mode="leetcode")
    return card


@app.post("/leetcode/codegen")
def leetcode_codegen(req: LeetCodeRequest):
    _require_session(req.session_id)
    # LeetCode is code-help (Qwen generates, DeepSeek explains), not subject
    # explainer -- it intentionally does not touch the book library. See
    # orchestrator.run_subject_turn / prompts.py for the same rule.
    try:
        brute_code, optimized_code = generate_leetcode_solutions(req.problem)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"brute_code": brute_code, "optimized_code": optimized_code}


@app.post("/leetcode/teach")
def leetcode_teach(req: LeetCodeTeachRequest):
    _require_session(req.session_id)
    try:
        card = teach_leetcode(req.problem, req.brute_code, req.optimized_code)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))
    memory_db.add_message(req.session_id, req.problem, card, mode="leetcode")
    return card


# --------------------------------------------------------------------------
# Workspace
# --------------------------------------------------------------------------
@app.post("/workspace/set")
def workspace_set(req: WorkspaceSetRequest):
    try:
        return workspace.set_workspace_root(req.path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/workspace/ingest")
def workspace_ingest():
    try:
        return workspace.ingest_workspace()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/workspace/status")
def workspace_status_endpoint():
    return workspace.workspace_status()


# --------------------------------------------------------------------------
# Background ingestion jobs
#
# Embedding a single large ebook can take tens of minutes (one HTTP call to
# Ollama per chunk), and a whole subject folder or "ingest everything" run
# can take hours. That's fundamentally incompatible with a single blocking
# HTTP request -- any reasonable client timeout will fire long before the
# work is done, even though the backend is still making real progress.
# Instead, ingestion runs in a background thread; the POST endpoint returns
# instantly with {"status": "started"}, and the frontend polls the
# matching GET .../status endpoint for live progress.
# --------------------------------------------------------------------------
_ingest_jobs: dict[str, dict] = {}
_ingest_jobs_lock = threading.Lock()


def _start_ingest_job(job_key: str, run_fn):
    with _ingest_jobs_lock:
        existing = _ingest_jobs.get(job_key)
        if existing and existing.get("state") == "running":
            return {"status": "already_running", "job": existing}
        job = {
            "state": "running",
            "started_at": time.time(),
            "finished_at": None,
            "progress": {},
            "result": None,
            "error": None,
        }
        _ingest_jobs[job_key] = job

    def _progress_cb(update: dict):
        with _ingest_jobs_lock:
            _ingest_jobs[job_key]["progress"] = update

    def _worker():
        try:
            result = run_fn(_progress_cb)
            with _ingest_jobs_lock:
                _ingest_jobs[job_key]["state"] = "done"
                _ingest_jobs[job_key]["result"] = result
                _ingest_jobs[job_key]["finished_at"] = time.time()
        except Exception as e:  # OllamaError or anything unexpected
            log.error(f"ingest job '{job_key}' failed: {e}")
            with _ingest_jobs_lock:
                _ingest_jobs[job_key]["state"] = "error"
                _ingest_jobs[job_key]["error"] = str(e)
                _ingest_jobs[job_key]["finished_at"] = time.time()

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started", "job": _ingest_jobs[job_key]}


def _ingest_job_status(job_key: str) -> dict:
    with _ingest_jobs_lock:
        job = _ingest_jobs.get(job_key)
    if not job:
        return {"state": "not_started"}
    return job


# --------------------------------------------------------------------------
# Subjects (Python, Git, DSA, OS, CN, DBMS, Software Engineering,
# System Design, Data Science, ML, DL, AI & LLMs, MLOps, DevOps, Cloud,
# Data Engineering, Interview Preparation)
#
# Each subject has its own isolated vector db (Books/<Subject>/*.pdf), its
# own manifest, and its own topic checklist.
# --------------------------------------------------------------------------
@app.get("/subjects")
def list_subjects():
    """All subjects with slug, book/chunk counts, and topic progress --
    everything the frontend needs to render the subject picker."""
    out = []
    for subject in SUBJECTS:
        status = subject_status(subject)
        progress = topics_module.progress_summary(subject)
        out.append({
            "subject": subject,
            "slug": subject_slug(subject),
            "total_files": status["total_files"],
            "total_chunks": status["total_chunks"],
            "topics_total": progress["total"],
            "topics_done": progress["done"],
        })
    return out


@app.post("/subjects/ingest_all")
def subjects_ingest_all():
    """Kick off a background job that ingests every subject folder under
    Books/ in sequence. Returns immediately -- poll /subjects/ingest_all/status
    for progress. This can legitimately take hours for a large library, so
    it must not be a single blocking HTTP request."""
    return _start_ingest_job("__all__", lambda cb: ingest_all_subjects(progress_cb=cb))


@app.get("/subjects/ingest_all/status")
def subjects_ingest_all_status():
    return _ingest_job_status("__all__")


@app.post("/subjects/{subject_slug_value}/ingest")
def subject_ingest(subject_slug_value: str):
    """Kick off a background job that ingests one subject. Returns
    immediately -- poll /subjects/{slug}/ingest/status for progress."""
    subject = _require_subject(subject_slug_value)
    return _start_ingest_job(subject_slug_value, lambda cb: ingest_subject(subject, progress_cb=cb))


@app.get("/subjects/{subject_slug_value}/ingest/status")
def subject_ingest_status(subject_slug_value: str):
    _require_subject(subject_slug_value)
    return _ingest_job_status(subject_slug_value)


@app.get("/subjects/{subject_slug_value}/status")
def subject_status_endpoint(subject_slug_value: str):
    subject = _require_subject(subject_slug_value)
    return subject_status(subject)


@app.get("/subjects/{subject_slug_value}/topics")
def subject_topics(subject_slug_value: str):
    """The checklist: every unique topic found across this subject's books,
    with its done/undone state."""
    subject = _require_subject(subject_slug_value)
    return {
        "subject": subject,
        "topics": topics_module.get_checklist(subject),
        "progress": topics_module.progress_summary(subject),
    }


@app.post("/subjects/{subject_slug_value}/topics/toggle")
def subject_topics_toggle(subject_slug_value: str, req: TopicToggleRequest):
    subject = _require_subject(subject_slug_value)
    try:
        checklist = topics_module.set_topic_done(subject, req.topic, req.done)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "subject": subject,
        "topics": checklist,
        "progress": topics_module.progress_summary(subject),
    }


@app.post("/subjects/{subject_slug_value}/chat")
def subject_chat(subject_slug_value: str, req: SubjectChatRequest):
    """Ask a question about this subject, grounded in that subject's own
    vector db (its books only)."""
    subject = _require_subject(subject_slug_value)
    _require_session(req.session_id)
    try:
        card = run_subject_turn(subject, req.question, mode_hint=req.mode_hint)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))
    memory_db.add_message(req.session_id, req.question, card, mode=f"subject:{subject_slug(subject)}")
    return card


@app.post("/subjects/{subject_slug_value}/explain_topic")
def explain_topic_endpoint(subject_slug_value: str, req: TopicExplainRequest):
    subject = _require_subject(subject_slug_value)
    _require_session(req.session_id)
    
    # Retrieve chunks directly matching the topic name
    chunks = retrieve(req.topic, top_k=RETRIEVAL_TOP_K, subject=subject)
    context = _format_context(chunks)
    
    # Run the explanation turn with a specialized prompt
    try:
        card = explain_topic_concepts(req.topic, context, subject=subject)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))
        
    card["detected_mode"] = "concept"
    card["subject"] = subject
    if not card.get("sources") and chunks:
        card["sources"] = [
            {"source": c["source"], "note": "retrieved from database matching topic"} for c in chunks
        ]
        
    # Save the message with the topic name as the question, utilizing topic-scoped mode
    memory_db.add_message(req.session_id, f"Explain topic: {req.topic}", card, mode=f"subject:{subject_slug(subject)}:{req.topic}")
    return card


@app.post("/subjects/{subject_slug_value}/followup")
def topic_followup_endpoint(subject_slug_value: str, req: TopicFollowupRequest):
    subject = _require_subject(subject_slug_value)
    _require_session(req.session_id)
    
    # 1. Fetch previous chat history for this topic
    mode_str = f"subject:{subject_slug(subject)}:{req.topic}"
    history_messages = memory_db.get_messages(req.session_id, mode=mode_str)
    
    # Format chat history for model context
    chat_history = []
    for msg in history_messages:
        # User prompt
        chat_history.append({"role": "user", "content": msg["question"]})
        # Assistant answer
        md_text = msg["card"].get("text") or msg["card"].get("concept") or msg["card"].get("what_is_it") or ""
        chat_history.append({"role": "assistant", "content": md_text})
        
    # 2. Retrieve chunks directly matching the topic name
    chunks = retrieve(req.topic, top_k=RETRIEVAL_TOP_K, subject=subject)
    context = _format_context(chunks)
    
    # 3. Call the specialized followup turn
    try:
        card = explain_topic_followup(req.topic, req.question, chat_history, context, subject=subject)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))
        
    card["detected_mode"] = "concept"
    card["subject"] = subject
    if not card.get("sources") and chunks:
        card["sources"] = [
            {"source": c["source"], "note": "retrieved from database matching topic"} for c in chunks
        ]
        
    # 4. Save the followup message
    memory_db.add_message(req.session_id, req.question, card, mode=mode_str)
    return card


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
@app.get("/session/{session_id}/export")
def export_pdf(session_id: int, mode: str | None = None):
    session = _require_session(session_id)
    messages = memory_db.get_messages(session_id, mode=mode)
    if not messages:
        raise HTTPException(status_code=400, detail="No messages found for this mode to export.")
    pdf_bytes = export_session_pdf(session["title"], messages)
    filename = f"study_notes_{mode.replace(':', '_') if mode else 'session'}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
