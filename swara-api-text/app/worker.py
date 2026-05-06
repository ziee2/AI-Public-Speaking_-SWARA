"""
RQ Worker
Run this to start the background worker
"""

import time
import sys
from rq import Worker
from app.core.redis_client import get_redis_connection, get_queue, check_redis_connection
from app.config import settings


def run_worker():
    """Run RQ worker with retry logic"""
    
    print(f"🔄 Starting RQ Worker...")
    print(f"📊 Queue: {settings.QUEUE_NAME}")
    print(f"🔗 Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    
    # Wait for Redis to be ready
    max_retries = 30
    retry_interval = 2
    
    for attempt in range(1, max_retries + 1):
        is_connected, error_msg = check_redis_connection()
        
        if is_connected:
            print(f"✅ Redis connected!")
            break
        else:
            if attempt < max_retries:
                print(f"⏳ Waiting for Redis... (attempt {attempt}/{max_retries})")
                time.sleep(retry_interval)
            else:
                print(f"❌ Failed to connect to Redis after {max_retries} attempts")
                print(f"   Error: {error_msg}")
                sys.exit(1)
    
    # Start worker
    redis_conn = get_redis_connection()
    queue = get_queue()
    
    print(f"🚀 Worker ready and listening for tasks!")
    print(f"📊 Queue length: {queue.count}\n")
    
    # Create worker with logging
    worker = Worker(
        [queue], 
        connection=redis_conn,
        log_job_description=True,  # Log job details
        name=f"worker-{settings.QUEUE_NAME}"  # Named worker for debugging
    )
    
    # Work with job logging
    worker.work(logging_level='INFO')


if __name__ == "__main__":
    run_worker()
