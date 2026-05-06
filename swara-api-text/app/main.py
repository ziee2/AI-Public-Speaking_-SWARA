"""
FastAPI Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.routes import router
from app.core.storage import ensure_upload_dir

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Audio Analysis API for Public Speaking Assessment"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["audio-analysis"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print(f"🚀 Starting {settings.APP_NAME} v{settings.VERSION}")
    ensure_upload_dir()
    print(f"✅ Upload directory ready: {settings.UPLOAD_DIR}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
