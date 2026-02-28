"""
Pipeline Orchestrator
======================
Runs the full analysis pipeline in sequence:
  1. Load messages from DB
  2. Score each contact
  3. Detect anomalies / drift
  4. Generate AI actions
  5. Optionally notify n8n webhooks

This is what n8n triggers on schedule.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Any
from app.core.db_helpers import (
    get_all_messages, save_contact, clear_actions, save_actions,
    save_pipeline_run, update_pipeline_run
)
from app.services.scoring_engine import score_contact
from app.services.action_generator import generate_actions_for_all
from app.services.n8n_client import notify_pipeline_complete, notify_new_actions


async def run_pipeline(trigger: str = "manual") -> Dict[str, Any]:
    """
    Full pipeline run. Returns summary dict.

    Args:
        trigger: 'manual' | 'scheduled' | 'webhook'
    """
    run_id = str(uuid.uuid4())[:8]
    started_at = datetime.now()

    print(f"\n{'='*50}")
    print(f"[PIPELINE] Run {run_id} started — trigger: {trigger}")
    print(f"{'='*50}")

    log_entry = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": None,
        "contacts_processed": 0,
        "actions_generated": 0,
        "status": "running",
        "trigger": trigger,
        "error": None,
    }
    await save_pipeline_run(log_entry)

    try:
        # ── STAGE 1: Score all contacts ─────────────────────────
        messages_dict = await get_all_messages()
        print(f"\n[STAGE 1] Scoring {len(messages_dict)} contacts...")
        scored_profiles = []

        for contact_name, messages in messages_dict.items():
            print(f"  → Scoring: {contact_name} ({len(messages)} msgs)")
            profile = score_contact(contact_name, messages)
            if profile:
                await save_contact(profile)
                scored_profiles.append(profile)

        print(f"  ✓ Scored {len(scored_profiles)} contacts")

        # ── STAGE 2: Generate AI Actions ────────────────────────
        print(f"\n[STAGE 2] Generating AI actions...")
        # Clear old pending actions
        await clear_actions()
        new_actions = generate_actions_for_all(scored_profiles)
        if new_actions:
            await save_actions(new_actions)
        print(f"  ✓ Generated {len(new_actions)} actions")

        # ── STAGE 3: Notify n8n ─────────────────────────────────
        print(f"\n[STAGE 3] Notifying n8n...")
        summary = _build_summary(scored_profiles, new_actions, run_id)
        notify_pipeline_complete(summary)
        if new_actions:
            notify_new_actions(new_actions)

        # ── Finalize ────────────────────────────────────────────
        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()

        updates = {
            "completed_at": completed_at,
            "contacts_processed": len(scored_profiles),
            "actions_generated": len(new_actions),
            "status": "completed",
            "duration_seconds": round(duration, 2),
        }
        await update_pipeline_run(run_id, updates)
        log_entry.update(updates)

        print(f"\n[PIPELINE] ✓ Complete in {duration:.1f}s")
        print(f"  Contacts: {len(scored_profiles)} | Actions: {len(new_actions)}")
        print(f"{'='*50}\n")

        return log_entry

    except Exception as e:
        updates = {
            "completed_at": datetime.now(),
            "status": "failed",
            "error": str(e),
        }
        await update_pipeline_run(run_id, updates)
        log_entry.update(updates)
        print(f"[PIPELINE] ✗ Failed: {e}")
        raise


def _build_summary(profiles: List[Dict], actions: List[Dict], run_id: str) -> Dict[str, Any]:
    ghosted    = [p for p in profiles if p["is_ghosted"]]
    drifting   = [p for p in profiles if p["drift_detected"]]
    avg_health = round(sum(p["health_score"] for p in profiles) / max(len(profiles), 1), 1)
    critical   = [a for a in actions if a["urgency"] == "CRITICAL"]

    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "total_contacts": len(profiles),
        "avg_health_score": avg_health,
        "ghosted_count": len(ghosted),
        "drifting_count": len(drifting),
        "actions_count": len(actions),
        "critical_actions": len(critical),
        "top_urgent": [
            {"name": a["contact_name"], "urgency": a["urgency"], "type": a["action_type"]}
            for a in actions[:3]
        ],
    }
