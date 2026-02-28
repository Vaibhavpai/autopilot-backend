"""
Relationship Health Scoring Engine
===================================
Composite score (0–100) from 4 weighted components:

  Recency Score    (30%) — how recently did they message?
  Frequency Score  (30%) — how often do they message per week?
  Response Ratio   (20%) — how often do they reply to you?
  Sentiment Score  (20%) — VADER compound sentiment of their messages

Drift Detection:
  Compares last 30 days activity vs previous 30 days.
  >50% drop = drift detected.

Tags: ACTIVE | CLOSE | STABLE | FADING | GHOSTED
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Scoring weights
W_RECENCY   = 0.30
W_FREQUENCY = 0.30
W_RESPONSE  = 0.20
W_SENTIMENT = 0.20

GHOSTED_THRESHOLD_DAYS  = 30
FADING_THRESHOLD_DAYS   = 14
DRIFT_DROP_THRESHOLD    = 0.50   # 50% drop triggers drift


def compute_recency_score(days_since: int) -> float:
    """Exponential decay. 0 days = 100, 30+ days = near 0."""
    if days_since == 0:
        return 100.0
    if days_since >= 60:
        return 0.0
    import math
    return round(100 * math.exp(-0.07 * days_since), 2)


def compute_frequency_score(messages: List[Dict], window_days: int = 30) -> float:
    """Messages per week over last N days, normalized to 0–100."""
    now = datetime.now()
    cutoff = now - timedelta(days=window_days)
    recent = [m for m in messages if m["timestamp"] >= cutoff]
    msgs_per_week = (len(recent) / window_days) * 7
    # Cap at 30 msgs/week = 100
    return round(min(100.0, (msgs_per_week / 30) * 100), 2)


def compute_response_ratio(messages: List[Dict]) -> float:
    """
    Ratio of contact replies to user messages.
    Scans sequential pairs — if user sends, check if contact replies within next 5 messages.
    """
    if not messages:
        return 0.5

    user_msgs = 0
    contact_replies = 0

    for i, msg in enumerate(messages):
        if msg["sender"] == "user":
            user_msgs += 1
            # Check if contact replied in next 5 messages
            window = messages[i+1:i+6]
            if any(m["sender"] != "user" for m in window):
                contact_replies += 1

    if user_msgs == 0:
        return 1.0
    return round(min(1.0, contact_replies / user_msgs), 3)


def compute_sentiment_score(messages: List[Dict]) -> float:
    """
    Average VADER compound sentiment of contact's messages.
    Returns -1 to +1. Neutral = 0.
    """
    contact_msgs = [m for m in messages if m["sender"] != "user"][-50:]  # last 50 msgs
    if not contact_msgs:
        return 0.0

    scores = [analyzer.polarity_scores(m["content"])["compound"] for m in contact_msgs]
    return round(sum(scores) / len(scores), 3)


def detect_drift(messages: List[Dict]) -> Tuple[bool, str]:
    """
    Compare last 30d vs prior 30d message count.
    Returns (drift_detected, severity)
    """
    now = datetime.now()
    last_30  = [m for m in messages if m["timestamp"] >= now - timedelta(days=30)]
    prior_30 = [m for m in messages if now - timedelta(days=60) <= m["timestamp"] < now - timedelta(days=30)]

    if not prior_30:
        return False, "none"

    drop = 1 - (len(last_30) / max(len(prior_30), 1))

    if drop >= 0.8:
        return True, "severe"
    elif drop >= 0.6:
        return True, "moderate"
    elif drop >= DRIFT_DROP_THRESHOLD:
        return True, "mild"
    return False, "none"


def is_ghosted(messages: List[Dict]) -> bool:
    """True if: last message was from user AND >30 days ago."""
    if not messages:
        return False
    sorted_msgs = sorted(messages, key=lambda x: x["timestamp"])
    last = sorted_msgs[-1]
    days_since = (datetime.now() - last["timestamp"]).days
    return last["sender"] == "user" and days_since > GHOSTED_THRESHOLD_DAYS


def compute_weekly_activity(messages: List[Dict], weeks: int = 12) -> List[Dict]:
    """Returns weekly message counts for the last N weeks."""
    now = datetime.now()
    result = []
    for i in range(weeks, 0, -1):
        week_start = now - timedelta(weeks=i)
        week_end   = now - timedelta(weeks=i-1)
        count = sum(1 for m in messages if week_start <= m["timestamp"] < week_end)
        week_label = week_start.strftime("W%U '%y")
        result.append({"week": week_label, "message_count": count})
    return result


def assign_tag(health_score: float, drift: bool, ghosted: bool,
               days_since: int, freq_score: float) -> str:
    if ghosted:
        return "GHOSTED"
    if health_score >= 80 and freq_score >= 70:
        return "CLOSE"
    if drift and health_score < 50:
        return "FADING"
    if health_score >= 60:
        return "ACTIVE" if freq_score >= 50 else "STABLE"
    if health_score >= 40:
        return "STABLE"
    return "FADING"


def assign_trend(messages: List[Dict]) -> str:
    """Compare last 14d vs prior 14d frequency."""
    now = datetime.now()
    last_14  = len([m for m in messages if m["timestamp"] >= now - timedelta(days=14)])
    prior_14 = len([m for m in messages if now - timedelta(days=28) <= m["timestamp"] < now - timedelta(days=14)])
    if prior_14 == 0:
        return "stable"
    ratio = last_14 / prior_14
    if ratio >= 1.2:
        return "up"
    if ratio <= 0.8:
        return "down"
    return "stable"


def score_contact(contact_name: str, messages: List[Dict]) -> Dict[str, Any]:
    """
    Full scoring pipeline for a single contact.
    Returns a dict matching ContactProfile schema.
    """
    if not messages:
        return {}

    sorted_msgs = sorted(messages, key=lambda x: x["timestamp"])
    last_msg = sorted_msgs[-1]
    days_since = (datetime.now() - last_msg["timestamp"]).days

    # Component scores
    recency   = compute_recency_score(days_since)
    frequency = compute_frequency_score(messages)
    response  = compute_response_ratio(sorted_msgs)
    sentiment = compute_sentiment_score(messages)

    # Normalize sentiment from [-1,1] to [0,100]
    sentiment_norm = (sentiment + 1) / 2 * 100

    # Composite health score
    health = (
        W_RECENCY   * recency        +
        W_FREQUENCY * frequency      +
        W_RESPONSE  * (response * 100) +
        W_SENTIMENT * sentiment_norm
    )
    health = round(min(100.0, max(0.0, health)), 1)

    drift, drift_severity = detect_drift(messages)
    ghosted = is_ghosted(sorted_msgs)
    tag     = assign_tag(health, drift, ghosted, days_since, frequency)
    trend   = assign_trend(messages)
    weekly  = compute_weekly_activity(messages)

    # Avatar initials
    parts   = contact_name.split()
    avatar  = (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else contact_name[:2].upper()

    # Detect last topic (last message that has >4 words)
    long_msgs = [m for m in sorted_msgs if m["sender"] != "user" and len(m["content"].split()) > 4]
    last_topic = long_msgs[-1]["content"][:80] if long_msgs else sorted_msgs[-1]["content"][:80]

    platform = sorted_msgs[-1].get("platform", "unknown")

    return {
        "contact_id": contact_name.lower().replace(" ", "_"),
        "name": contact_name,
        "handle": f"@{contact_name.split()[0].lower()}",
        "platform": platform,
        "avatar": avatar,
        "total_messages": len(messages),
        "last_message_at": last_msg["timestamp"],
        "days_since_last_message": days_since,
        "recency_score": recency,
        "frequency_score": frequency,
        "response_ratio": response,
        "sentiment_avg": sentiment,
        "health_score": health,
        "drift_detected": drift,
        "is_ghosted": ghosted,
        "drift_severity": drift_severity,
        "last_topic": last_topic,
        "tag": tag,
        "trend": trend,
        "weekly_activity": weekly,
    }
