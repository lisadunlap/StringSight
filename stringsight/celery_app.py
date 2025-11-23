from celery import Celery
from stringsight.config import settings

celery_app = Celery(
    "stringsight",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["stringsight.workers.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task settings
    task_track_started=True,
    task_time_limit=3600 * 24,  # 24 hours
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks to prevent memory leaks
)
