from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.core.database import pipeline_log, messages_db
from app.services.pipeline import run_pipeline

router = APIRouter()


@router.post("/run")
def trigger_pipeline(background_tasks: BackgroundTasks, trigger: str = "manual"):
    """
    Trigger a full pipeline run.
    Also called by n8n on schedule via POST /api/pipeline/run
    """
    if not messages_db:
        raise HTTPException(
            status_code=400,
            detail="No data loaded. Use /api/ingest/synthetic or upload a file first."
        )
    background_tasks.add_task(run_pipeline, trigger)
    return {"status": "Pipeline started", "trigger": trigger}


@router.post("/run/sync")
def trigger_pipeline_sync(trigger: str = "manual"):
    """Synchronous pipeline run (blocks until complete). Use for testing."""
    if not messages_db:
        raise HTTPException(status_code=400, detail="No data loaded.")
    result = run_pipeline(trigger)
    return {"status": "completed", "result": result}


@router.get("/status")
def pipeline_status():
    """Get last pipeline run status."""
    if not pipeline_log:
        return {"status": "never_run", "message": "Pipeline has not been run yet."}
    last = pipeline_log[-1]
    return last


@router.get("/history")
def pipeline_history():
    """Full pipeline run history."""
    return {"runs": list(reversed(pipeline_log)), "total": len(pipeline_log)}
