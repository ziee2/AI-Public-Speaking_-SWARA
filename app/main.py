import json
import uuid
import threading
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.audio_analysis import analyze_audio

app = FastAPI(
    title="SWARA Audio Analysis API",
    description="API for analyzing public speaking audio recordings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task store: task_id -> {"status": str, "result": dict|None, "error": str|None}
_tasks: dict = {}
_tasks_lock = threading.Lock()


def _run_analysis(
    task_id: str,
    audio_bytes: bytes,
    custom_topic: Optional[str],
    reference_text: Optional[str],
    custom_keywords: Optional[list],
):
    try:
        result = analyze_audio(
            audio_bytes=audio_bytes,
            custom_topic=custom_topic,
            reference_text=reference_text,
            custom_keywords=custom_keywords,
        )
        with _tasks_lock:
            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["result"] = result
    except Exception as exc:
        with _tasks_lock:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = str(exc)


@app.post("/api/v1/analyze")
async def submit_analysis(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    custom_topic: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(None),
    custom_keywords: Optional[str] = Form(None),
):
    """
    Submit an audio file for analysis.

    - **audio**: WAV audio file
    - **custom_topic**: Optional topic string for keyword relevance scoring
    - **reference_text**: Optional reference/script text for structure scoring
    - **custom_keywords**: Optional JSON array of keywords (e.g. '["kata1","kata2"]')

    Returns a **task_id** that can be polled via GET /api/v1/status/{task_id}.
    """
    audio_bytes = await audio.read()

    keywords_list: Optional[list] = None
    if custom_keywords:
        try:
            keywords_list = json.loads(custom_keywords)
            if not isinstance(keywords_list, list):
                keywords_list = None
        except (json.JSONDecodeError, ValueError):
            keywords_list = None

    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {"status": "pending", "result": None, "error": None}

    background_tasks.add_task(
        _run_analysis,
        task_id,
        audio_bytes,
        custom_topic or None,
        reference_text or None,
        keywords_list,
    )

    return {"task_id": task_id}


@app.get("/api/v1/status/{task_id}")
async def get_status(task_id: str):
    """
    Poll the status and result of a previously submitted analysis task.

    Returns:
    - **status**: "pending" | "completed" | "failed"
    - **result**: analysis result object when completed, otherwise null
    - **error**: error message when failed, otherwise null
    """
    with _tasks_lock:
        task = _tasks.get(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "status": task["status"],
        "result": task["result"],
        "error": task["error"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
