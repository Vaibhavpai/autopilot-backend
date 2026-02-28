from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.core.db_helpers import count_messages, get_last_pipeline_run, get_pipeline_history
from app.services.pipeline import run_pipeline

router = APIRouter()


@router.post("/run")
async def trigger_pipeline(background_tasks: BackgroundTasks, trigger: str = "manual"):
    """
    Trigger a full pipeline run.
    Also called by n8n on schedule via POST /api/pipeline/run
    """
    msg_count = await count_messages()
    if msg_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No data loaded. Use /api/ingest/synthetic or upload a file first."
        )
    background_tasks.add_task(run_pipeline, trigger)
    return {"status": "Pipeline started", "trigger": trigger}


@router.post("/run/sync")
async def trigger_pipeline_sync(trigger: str = "manual"):
    """Synchronous pipeline run (blocks until complete). Use for testing."""
    msg_count = await count_messages()
    if msg_count == 0:
        raise HTTPException(status_code=400, detail="No data loaded.")
    result = await run_pipeline(trigger)
    return {"status": "completed", "result": result}


@router.get("/status")
async def pipeline_status():
    """Get last pipeline run status."""
    last = await get_last_pipeline_run()
    if not last:
        return {"status": "never_run", "message": "Pipeline has not been run yet."}
    return last


@router.get("/history")
async def pipeline_history():
    """Full pipeline run history."""
    runs = await get_pipeline_history()
    return {"runs": runs, "total": len(runs)}
