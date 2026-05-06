# Hugging Face Spaces - Single Container Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including Redis
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    redis-server \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory for models BEFORE copying code
# This ensures model downloads are cached even when code changes
RUN mkdir -p /.cache && chmod -R 777 /.cache
RUN mkdir -p /data/.cache && chmod -R 777 /data/.cache
ENV HF_HOME=/.cache
ENV TORCH_HOME=/data/.cache
ENV XDG_CACHE_HOME=/.cache
ENV WHISPER_CACHE=/data/.cache

# Pre-download models during build (HF Pro with persistent storage)
# These layers will be CACHED and won't rebuild when only code changes

# 1. Download Structure Model from HF Hub (~475MB)
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    print('📥 Downloading Structure Model from HF Hub...'); \
    AutoTokenizer.from_pretrained('Cyberlace/swara-structure-model', cache_dir='/.cache'); \
    AutoModelForSequenceClassification.from_pretrained('Cyberlace/swara-structure-model', cache_dir='/.cache'); \
    print('✅ Structure Model cached!')" && chmod -R 777 /.cache

# 2. Whisper medium: LAZY LOADING on first request
# Build OOM - HF Space build container has RAM limit
# Will download to /data/.cache on FIRST REQUEST (~2-3 min)
# With HF Pro persistent storage, download persists across restarts
# Subsequent requests will be fast using cached model

# 3. Download Sentence Transformer for Keywords (~420MB)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    print('📥 Downloading Sentence Transformer...'); \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder='/.cache'); \
    print('✅ Sentence Transformer cached!')" && chmod -R 777 /.cache

# 4. Download Silero VAD (~10MB)
RUN python -c "import torch; \
    print('📥 Downloading Silero VAD model...'); \
    torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False); \
    print('✅ Silero VAD cached!')" && chmod -R 777 /.cache

# Copy application code LAST (after model downloads)
# This way, code changes don't invalidate model cache layers
COPY . .

# Fix all cache permissions after code copy (in case any new cache created)
RUN chmod -R 777 /.cache || true

# Create uploads directory with proper permissions
RUN mkdir -p uploads && chmod 777 uploads

# Make start script executable
RUN chmod +x start.sh

# Expose Hugging Face Spaces port
EXPOSE 7860

# Start script (Redis + Worker + API)
CMD ["./start.sh"]
