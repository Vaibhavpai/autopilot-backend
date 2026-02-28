"""
AI Action Generator
====================
Uses Claude API to generate personalized re-engagement messages
and follow-up suggestions based on relationship context.
"""
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import anthropic

from app.core.config import settings

client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def _build_prompt(profile: Dict[str, Any]) -> str:
    return f"""You are a social intelligence assistant helping a Gen Z user maintain their friendships.

Here is the relationship context:
- Contact name: {profile['name']}
- Days since last message: {profile['days_since_last_message']}
- Health score: {profile['health_score']}/100
- Status: {profile['tag']}
- Drift detected: {profile['drift_detected']} (severity: {profile['drift_severity']})
- Ghosted (user sent last, no reply): {profile['is_ghosted']}
- Last conversation topic: "{profile['last_topic']}"
- Platform: {profile['platform']}
- Total messages exchanged: {profile['total_messages']}
- Sentiment of recent messages: {profile['sentiment_avg']:.2f} (-1 negative, 0 neutral, +1 positive)

Task: Generate a short, natural, casual re-engagement message that:
1. Feels authentic and warm, not robotic
2. References the last topic naturally if relevant
3. Is appropriate for Gen Z texting style (casual, maybe an emoji)
4. Is under 30 words
5. Opens a conversation naturally

Also provide:
- action_type: one of [RE-ENGAGE, FOLLOW-UP, CHECK-IN]
- urgency: one of [CRITICAL, HIGH, MEDIUM, LOW]
- reason: one sentence explaining why this action is recommended

Respond in this exact JSON format:
{{
  "action_type": "...",
  "urgency": "...",
  "suggested_message": "...",
  "reason": "..."
}}"""


def generate_action_for_contact(profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate a single AI action suggestion for a contact."""
    # Only generate actions for contacts that need attention
    if profile["health_score"] >= 80 and not profile["drift_detected"]:
        return None

    if profile["tag"] not in ("FADING", "GHOSTED", "STABLE") and profile["days_since_last_message"] < 7:
        return None

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": _build_prompt(profile)}]
        )

        import json
        text = response.content[0].text.strip()
        # Strip markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)

        return {
            "action_id": str(uuid.uuid4()),
            "contact_id": profile["contact_id"],
            "contact_name": profile["name"],
            "action_type": data.get("action_type", "RE-ENGAGE"),
            "urgency": data.get("urgency", "MEDIUM"),
            "suggested_message": data.get("suggested_message", ""),
            "reason": data.get("reason", ""),
            "created_at": datetime.now(),
            "status": "pending",
        }

    except Exception as e:
        print(f"  ⚠ Claude API error for {profile['name']}: {e}")
        # Fallback rule-based action
        return _fallback_action(profile)


def _fallback_action(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based fallback when Claude API is unavailable."""
    days = profile["days_since_last_message"]
    name = profile["name"].split()[0]

    if profile["is_ghosted"]:
        msg = f"hey {name}, all good? been a minute 👋"
        urgency = "CRITICAL"
    elif days > 20:
        topic = profile["last_topic"][:30] if profile["last_topic"] else "everything"
        msg = f"yo {name}! been ages — how'd {topic[:20]} go?"
        urgency = "HIGH"
    else:
        msg = f"hey {name}, what's good lately?"
        urgency = "MEDIUM"

    return {
        "action_id": str(uuid.uuid4()),
        "contact_id": profile["contact_id"],
        "contact_name": profile["name"],
        "action_type": "RE-ENGAGE",
        "urgency": urgency,
        "suggested_message": msg,
        "reason": f"No interaction for {days} days. Relationship showing signs of drift.",
        "created_at": datetime.now(),
        "status": "pending",
    }


def generate_actions_for_all(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run action generation across all contacts. Returns list of actions."""
    actions = []
    # Prioritize by urgency: ghosted first, then by days_since
    sorted_profiles = sorted(
        profiles,
        key=lambda p: (not p["is_ghosted"], p["health_score"])
    )
    for profile in sorted_profiles:
        action = generate_action_for_contact(profile)
        if action:
            actions.append(action)
            print(f"  ✓ Action generated for {profile['name']} [{action['urgency']}]")
    return actions
