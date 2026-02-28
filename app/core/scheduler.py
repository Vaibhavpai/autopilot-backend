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
    import asyncio
    from app.core.db_helpers import count_messages
    
    async def _run():
        msg_count = await count_messages()
        if msg_count == 0:
            print("[SCHEDULER] Skipping run — no data loaded.")
            return
        print(f"[SCHEDULER] Auto-run triggered at {datetime.now().isoformat()}")
        from app.services.pipeline import run_pipeline
        await run_pipeline(trigger="scheduled")
    
    # Run async function in sync context
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(_run())


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
