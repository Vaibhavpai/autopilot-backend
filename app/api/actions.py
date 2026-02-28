from fastapi import APIRouter, HTTPException
from typing import Optional
from app.core.database import actions_db

router = APIRouter()


@router.get("/")
def get_actions(
    status: Optional[str] = None,
    urgency: Optional[str] = None,
):
    """Get all AI-generated action suggestions."""
    actions = list(actions_db)
    if status:
        actions = [a for a in actions if a.get("status") == status]
    if urgency:
        actions = [a for a in actions if a.get("urgency") == urgency.upper()]

    # Sort: CRITICAL first, then HIGH, MEDIUM, LOW
    urgency_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    actions.sort(key=lambda a: urgency_order.get(a.get("urgency", "LOW"), 4))

    return {"actions": actions, "total": len(actions)}


@router.patch("/{action_id}/status")
def update_action_status(action_id: str, status: str):
    """
    Update action status: pending → sent | dismissed.
    Called when user marks 'Sent' or dismisses from dashboard.
    """
    if status not in ("sent", "dismissed", "pending"):
        raise HTTPException(status_code=400, detail="status must be: sent | dismissed | pending")

    for action in actions_db:
        if action["action_id"] == action_id:
            action["status"] = status
            return {"success": True, "action_id": action_id, "new_status": status}

    raise HTTPException(status_code=404, detail=f"Action '{action_id}' not found")


@router.delete("/{action_id}")
def delete_action(action_id: str):
    """Remove a specific action."""
    for i, action in enumerate(actions_db):
        if action["action_id"] == action_id:
            actions_db.pop(i)
            return {"success": True, "deleted": action_id}
    raise HTTPException(status_code=404, detail=f"Action '{action_id}' not found")
