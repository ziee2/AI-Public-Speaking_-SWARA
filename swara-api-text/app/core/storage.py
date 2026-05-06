"""
File storage utilities
"""

import os
import shutil
from pathlib import Path
from app.config import settings


def ensure_upload_dir():
    """Ensure upload directory exists"""
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """
    Save uploaded file
    
    Returns:
        str: Path to saved file
    """
    ensure_upload_dir()
    
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    return file_path


def delete_file(file_path: str):
    """Delete file if exists"""
    if os.path.exists(file_path):
        os.remove(file_path)


def cleanup_old_files(max_age_hours: int = 24):
    """Cleanup old uploaded files"""
    import time
    
    if not os.path.exists(settings.UPLOAD_DIR):
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(settings.UPLOAD_DIR):
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            
            if file_age > max_age_seconds:
                try:
                    delete_file(file_path)
                    print(f"Deleted old file: {filename}")
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")
