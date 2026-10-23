"""Production WSGI server settings:   gunicorn -c gunicorn.conf.py app:app

IMPORTANT - exactly ONE worker process:
  the analysis lock, the live progress log (/progress) and the loaded models live in the memory of the process. Several worker processes would each load the
  (large) models and /progress could be answered by a process that is not running the analysis. Concurrency comes from threads instead
  (the progress/health requests are served while an analysis runs in another thread).
"""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"      # hosting platforms inject PORT
workers = 1
worker_class = "gthread"
threads = int(os.environ.get("WEB_THREADS", "8"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "900"))   # a video analysis is one long request; raise this for long videos (0 = no limit)
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
