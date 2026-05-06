"""
Configuration file
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # App Info
    APP_NAME: str = "Swara API - Audio Analysis"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    
    # RQ Worker Configuration
    QUEUE_NAME: str = "audio_analysis"
    JOB_TIMEOUT: int = 3600  # 1 hour
    RESULT_TTL: int = 86400  # 24 hours
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100 MB
    ALLOWED_EXTENSIONS: list = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    
    # Model Configuration
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "medium")  # Changed to medium for better accuracy
    KATA_KUNCI_PATH: str = os.getenv("KATA_KUNCI_PATH", "./kata_kunci.json")
    
    # Device Configuration (CPU/GPU)
    DEVICE: str = os.getenv("DEVICE", "auto")  # 'auto', 'cpu', or 'cuda'
    
    # CORS
    CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"


settings = Settings()
