"""
Redis client setup
"""

import redis
from rq import Queue
from app.config import settings


def get_redis_connection():
    """Get Redis connection for RQ (without decode_responses)"""
    redis_kwargs = {
        'host': settings.REDIS_HOST,
        'port': settings.REDIS_PORT,
        'db': settings.REDIS_DB,
        'socket_connect_timeout': 5,  # Connection timeout
        'socket_timeout': 5,  # Socket timeout
        'retry_on_timeout': True,  # Retry on timeout
        'health_check_interval': 30,  # Health check every 30s
    }
    
    # Only add password if it's set
    if settings.REDIS_PASSWORD:
        redis_kwargs['password'] = settings.REDIS_PASSWORD
    
    # Don't use decode_responses for RQ compatibility
    return redis.Redis(**redis_kwargs)


def get_queue():
    """Get RQ Queue"""
    conn = get_redis_connection()
    return Queue(settings.QUEUE_NAME, connection=conn)


def check_redis_connection():
    """
    Check if Redis connection is working
    Returns tuple: (is_connected: bool, error_message: str)
    """
    try:
        conn = get_redis_connection()
        conn.ping()
        return True, None
    except redis.ConnectionError as e:
        return False, f"Redis connection error: {str(e)}"
    except Exception as e:
        return False, f"Redis error: {str(e)}"
