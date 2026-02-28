from fastapi import APIRouter, HTTPException
from typing import Optional
from app.core.db_helpers import get_all_actions, update_action_status, delete_action

router = APIRouter()


@router.get("/")
async def get_actions(
    status: Optional[str] = None,
    urgency: Optional[str] = None,
):
    """Get all AI-generated action suggestions."""
    actions = await get_all_actions()
    if status:
        actions = [a for a in actions if a.get("status") == status]
    if urgency:
        actions = [a for a in actions if a.get("urgency") == urgency.upper()]

    # Sort: CRITICAL first, then HIGH, MEDIUM, LOW
    urgency_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    actions.sort(key=lambda a: urgency_order.get(a.get("urgency", "LOW"), 4))

    return {"actions": actions, "total": len(actions)}


@router.patch("/{action_id}/status")
async def update_action_status_endpoint(action_id: str, status: str):
    """
    Update action status: pending → sent | dismissed.
    Called when user marks 'Sent' or dismisses from dashboard.
    """
    if status not in ("sent", "dismissed", "pending"):
        raise HTTPException(status_code=400, detail="status must be: sent | dismissed | pending")

    updated = await update_action_status(action_id, status)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Action '{action_id}' not found")
    
    return {"success": True, "action_id": action_id, "new_status": status}


@router.delete("/{action_id}")
async def delete_action_endpoint(action_id: str):
    """Remove a specific action."""
    deleted = await delete_action(action_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Action '{action_id}' not found")
    return {"success": True, "deleted": action_id}
