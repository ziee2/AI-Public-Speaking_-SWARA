#!/bin/bash

echo "=========================================="
echo "Starting Swara API Services"
echo "=========================================="

# Fix cache permissions on startup (in case of permission issues)
chmod -R 777 /.cache 2>/dev/null || true
chmod -R 777 /data/.cache 2>/dev/null || true

# Fix OpenMP warning - set proper thread count
export OMP_NUM_THREADS=4

# Set environment variables for Redis (localhost since in same container)
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0

# Start Redis in background with persistence DISABLED (in-memory only)
echo "[1/4] Starting Redis server (in-memory mode)..."
redis-server --daemonize yes --bind 127.0.0.1 --port 6379 \
    --save "" \
    --appendonly no \
    --maxmemory 512mb \
    --maxmemory-policy allkeys-lru

# Wait for Redis to be ready with timeout
echo "[2/4] Waiting for Redis to be ready..."
REDIS_TIMEOUT=30
ELAPSED=0
until redis-cli -h localhost -p 6379 ping 2>/dev/null | grep -q PONG; do
  if [ $ELAPSED -ge $REDIS_TIMEOUT ]; then
    echo "ERROR: Redis failed to start within ${REDIS_TIMEOUT}s"
    exit 1
  fi
  echo "   Waiting for Redis... (${ELAPSED}s)"
  sleep 2
  ELAPSED=$((ELAPSED + 2))
done

echo "   ✓ Redis is ready!"

# Start RQ worker in background
echo "[3/4] Starting RQ Worker..."
python -m app.worker &
WORKER_PID=$!
echo "   ✓ Worker started (PID: $WORKER_PID)"

# Give worker time to initialize
sleep 2

# Start FastAPI application
echo "[4/4] Starting FastAPI application..."
echo "=========================================="
echo "API will be available at:"
echo "  http://localhost:7860"
echo "  http://localhost:7860/docs (API Documentation)"
echo "=========================================="

uvicorn app.main:app --host 0.0.0.0 --port 7860
