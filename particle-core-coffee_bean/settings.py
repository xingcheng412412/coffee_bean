import os

CACHE_RESULT = os.environ.get("CACHE_RESULT", "1") == "1"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "16"))
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://:audit123456@127.0.0.1:6379/7")
CELERY_WORKER_CONCURRENCY = int(os.environ.get("CELERY_WORKER_CONCURRENCY", 2))
