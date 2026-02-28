from fastapi import APIRouter, HTTPException
from typing import List, Optional
from app.core.db_helpers import get_all_contacts, get_contact

router = APIRouter()


@router.get("/")
async def get_all_contacts_endpoint(
    tag: Optional[str] = None,
    min_score: Optional[float] = None,
    platform: Optional[str] = None,
):
    """
    Get all scored contact profiles.
    Optional filters: tag, min_score, platform
    """
    profiles = await get_all_contacts()

    if tag:
        profiles = [p for p in profiles if p.get("tag") == tag.upper()]
    if min_score is not None:
        profiles = [p for p in profiles if p.get("health_score", 0) >= min_score]
    if platform:
        profiles = [p for p in profiles if p.get("platform") == platform.lower()]

    # Sort by health score ascending (worst first — most needing attention)
    profiles.sort(key=lambda p: p.get("health_score", 0))
    return {"contacts": profiles, "total": len(profiles)}


@router.get("/summary")
async def get_summary():
    """Dashboard summary stats."""
    profiles = await get_all_contacts()
    if not profiles:
        return {"message": "No contacts scored yet. Run the pipeline first."}

    return {
        "total": len(profiles),
        "avg_health_score": round(sum(p["health_score"] for p in profiles) / len(profiles), 1),
        "ghosted_count": sum(1 for p in profiles if p.get("is_ghosted")),
        "drifting_count": sum(1 for p in profiles if p.get("drift_detected")),
        "by_tag": {
            tag: sum(1 for p in profiles if p.get("tag") == tag)
            for tag in ["ACTIVE", "CLOSE", "STABLE", "FADING", "GHOSTED"]
        },
        "top_at_risk": [
            {"name": p["name"], "score": p["health_score"], "tag": p["tag"]}
            for p in sorted(profiles, key=lambda x: x["health_score"])[:3]
        ],
    }


@router.get("/{contact_id}")
async def get_contact_endpoint(contact_id: str):
    """Get single contact profile by ID."""
    profile = await get_contact(contact_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Contact '{contact_id}' not found")
    return profile
