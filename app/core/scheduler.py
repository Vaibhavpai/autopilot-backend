"""
Scheduler — runs the pipeline automatically every 6 hours.
Attach to FastAPI lifespan in main.py when ready.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime

scheduler = BackgroundScheduler()


def scheduled_pipeline_run():
    """Called by APScheduler on interval."""
    from app.core.database import messages_db
    if not messages_db:
        print("[SCHEDULER] Skipping run — no data loaded.")
        return
    print(f"[SCHEDULER] Auto-run triggered at {datetime.now().isoformat()}")
    from app.services.pipeline import run_pipeline
    run_pipeline(trigger="scheduled")


def start_scheduler():
    scheduler.add_job(
        scheduled_pipeline_run,
        trigger=IntervalTrigger(hours=6),
        id="pipeline_auto_run",
        name="Autopilot Pipeline",
        replace_existing=True,
    )
    scheduler.start()
    print("[SCHEDULER] Started — pipeline will run every 6 hours.")


def stop_scheduler():
    scheduler.shutdown()
    print("[SCHEDULER] Stopped.")
