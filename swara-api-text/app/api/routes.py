"""
API Routes
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
import uuid
import os
import json

from app.models import TaskResponse, TaskStatusResponse, TaskStatus, AnalysisRequest
from app.core.redis_client import get_queue
from app.core.storage import save_uploaded_file
from app.config import settings
from app.tasks import process_audio_task

router = APIRouter()


@router.post("/analyze", response_model=TaskResponse)
async def analyze_audio(
    audio: UploadFile = File(...),
    reference_text: Optional[str] = Form(None),
    topic_id: Optional[str] = Form(None),
    custom_topic: Optional[str] = Form(None),
    custom_keywords: Optional[str] = Form(None),  # JSON string dari frontend
    analyze_tempo: bool = Form(True),
    analyze_articulation: bool = Form(True),
    analyze_structure: bool = Form(True),
    analyze_keywords: bool = Form(True),  # Default TRUE
    analyze_profanity: bool = Form(True)  # Default TRUE
):
    """
    Submit audio file untuk analisis
    
    Parameters:
    - audio: File audio (.wav, .mp3, .m4a, .flac, .ogg)
    - reference_text: Teks referensi untuk artikulasi (optional)
    - topic_id: ID topik dari database untuk Level 1-2 (optional)
    - custom_topic: Topik custom untuk Level 3 (optional)
    - custom_keywords: JSON array kata kunci dari GPT, contoh: ["inovasi", "kreativitas", "perubahan"] (optional)
    - analyze_tempo: Analisis tempo (default: true)
    - analyze_articulation: Analisis artikulasi (default: true)
    - analyze_structure: Analisis struktur (default: true)
    - analyze_keywords: Analisis kata kunci (default: true) - otomatis skip jika tidak ada topic_id/custom_keywords
    - analyze_profanity: Deteksi kata tidak senonoh (default: true)
    
    Returns task_id yang bisa digunakan untuk check status
    """
    
    # Generate task_id first
    task_id = str(uuid.uuid4())
    print(f"\n📥 [REQUEST] New request received - Task ID: {task_id}")
    print(f"   Filename: {audio.filename}")
    
    # Validate file extension
    file_ext = os.path.splitext(audio.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Validate file size
    content = await audio.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB"
        )
    
    # Parse custom_keywords dari JSON string
    parsed_custom_keywords = None
    if custom_keywords:
        try:
            parsed_custom_keywords = json.loads(custom_keywords)
            if not isinstance(parsed_custom_keywords, list):
                raise ValueError("custom_keywords harus berupa array")
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="custom_keywords harus berupa JSON array valid, contoh: [\"kata1\", \"kata2\"]"
            )
    
    # Save file
    filename = f"{task_id}{file_ext}"
    file_path = save_uploaded_file(content, filename)
    print(f"💾 [SAVE] File saved: {file_path}")
    
    # Submit task to queue
    queue = get_queue()
    job = queue.enqueue(
        process_audio_task,
        audio_path=file_path,
        reference_text=reference_text,
        topic_id=topic_id,
        custom_topic=custom_topic,
        custom_keywords=parsed_custom_keywords,
        analyze_tempo=analyze_tempo,
        analyze_articulation=analyze_articulation,
        analyze_structure=analyze_structure,
        analyze_keywords=analyze_keywords,
        analyze_profanity=analyze_profanity,
        job_id=task_id,
        job_timeout=settings.JOB_TIMEOUT,
        result_ttl=settings.RESULT_TTL,
        at_front=False  # CRITICAL: Ensure FIFO order, prevent job overwriting
    )
    
    print(f"✅ [QUEUED] Job enqueued - Job ID: {job.id}, Queue position: {queue.count}")
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        message="Task submitted successfully"
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Check status dari task
    
    Returns status dan result jika sudah selesai
    """
    from rq.job import Job
    from app.core.redis_client import get_redis_connection
    
    try:
        redis_conn = get_redis_connection()
        job = Job.fetch(task_id, connection=redis_conn)
        
        # Map job status to our TaskStatus
        if job.is_queued:
            status = TaskStatus.QUEUED
        elif job.is_started:
            status = TaskStatus.PROCESSING
        elif job.is_finished:
            status = TaskStatus.COMPLETED
        elif job.is_failed:
            status = TaskStatus.FAILED
        else:
            status = TaskStatus.QUEUED
        
        # Get result if completed
        result = None
        error = None
        
        if job.is_finished:
            job_result = job.result
            if isinstance(job_result, dict):
                if job_result.get('status') == 'completed':
                    result = job_result.get('result')
                elif job_result.get('status') == 'failed':
                    error = job_result.get('error')
                    status = TaskStatus.FAILED
        
        if job.is_failed:
            error = str(job.exc_info)
        
        return TaskStatusResponse(
            task_id=task_id,
            status=status,
            result=result,
            error=error,
            created_at=job.created_at.isoformat() if job.created_at else None,
            updated_at=job.ended_at.isoformat() if job.ended_at else None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Task not found: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    from app.core.redis_client import check_redis_connection
    from app.core.device import get_device_info
    
    is_connected, error_msg = check_redis_connection()
    
    if is_connected:
        redis_status = "healthy"
    else:
        redis_status = f"unhealthy: {error_msg}"
    
    # Get device information
    device_info = get_device_info()
    
    return {
        "status": "healthy" if is_connected else "degraded",
        "redis": redis_status,
        "version": settings.VERSION,
        "device": device_info
    }
