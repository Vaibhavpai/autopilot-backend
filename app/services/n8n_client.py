"""
n8n Webhook Client
===================
Sends structured payloads to n8n workflows.
n8n then handles: email digests, reminders, Slack pings etc.
"""
import httpx
from typing import List, Dict, Any
from app.core.config import settings


def _post(url: str, payload: Dict[str, Any]) -> bool:
    """Fire-and-forget POST to n8n webhook."""
    try:
        r = httpx.post(url, json=payload, timeout=5.0)
        r.raise_for_status()
        print(f"  ✓ n8n notified: {url} → {r.status_code}")
        return True
    except httpx.ConnectError:
        print(f"  ⚠ n8n not reachable at {url} (is n8n running?)")
        return False
    except Exception as e:
        print(f"  ⚠ n8n webhook error: {e}")
        return False


def notify_pipeline_complete(summary: Dict[str, Any]) -> bool:
    """
    Trigger n8n after pipeline completes.
    n8n can: send email digest, update dashboard stats, log to Notion, etc.
    """
    payload = {
        "event": "pipeline_complete",
        "data": summary,
    }
    return _post(settings.N8N_WEBHOOK_URL, payload)


def notify_new_actions(actions: List[Dict[str, Any]]) -> bool:
    """
    Send pending actions to n8n for reminder emails.
    n8n can: send email with action cards, push notification, Slack message.
    """
    critical = [a for a in actions if a["urgency"] in ("CRITICAL", "HIGH")]
    if not critical:
        return True

    payload = {
        "event": "new_actions",
        "data": {
            "total_actions": len(actions),
            "critical_count": len(critical),
            "actions": [
                {
                    "contact": a["contact_name"],
                    "type": a["action_type"],
                    "urgency": a["urgency"],
                    "message": a["suggested_message"],
                    "reason": a["reason"],
                }
                for a in critical[:5]  # top 5 only
            ],
        }
    }
    return _post(settings.N8N_REMINDER_WEBHOOK, payload)
