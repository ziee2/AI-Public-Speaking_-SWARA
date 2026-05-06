"""
Pydantic models untuk request/response
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enum"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisRequest(BaseModel):
    """Request untuk analisis audio"""
    reference_text: Optional[str] = Field(None, description="Teks referensi untuk perbandingan")
    topic_id: Optional[str] = Field(None, description="ID topik untuk analisis kata kunci")
    analyze_tempo: bool = Field(True, description="Analisis tempo dan jeda")
    analyze_articulation: bool = Field(True, description="Analisis artikulasi/pronunciation")
    analyze_structure: bool = Field(True, description="Analisis struktur berbicara")
    analyze_keywords: bool = Field(False, description="Analisis kata kunci (perlu topic_id)")


class TaskResponse(BaseModel):
    """Response untuk submit task"""
    task_id: str
    status: TaskStatus
    message: str


class TaskStatusResponse(BaseModel):
    """Response untuk check status task"""
    task_id: str
    status: TaskStatus
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AnalysisResult(BaseModel):
    """Full analysis result"""
    task_id: str
    status: str
    transcript: str
    
    # Results dari masing-masing analisis
    tempo: Optional[Dict[str, Any]] = None
    articulation: Optional[Dict[str, Any]] = None
    structure: Optional[Dict[str, Any]] = None
    keywords: Optional[Dict[str, Any]] = None
    
    # Summary
    overall_score: Optional[float] = None
    processing_time: Optional[float] = None
