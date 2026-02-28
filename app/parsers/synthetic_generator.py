"""
Generates realistic synthetic chat data for demo purposes.
Creates 8 contacts with varied relationship health patterns.
"""
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
from faker import Faker

fake = Faker()
random.seed(42)

# Define contact archetypes
ARCHETYPES = [
    {
        "name": "Alex Chen",
        "pattern": "active",          # messages daily
        "days_since": 1,
        "total_msgs": 380,
        "topics": ["hackathon collab", "late night debugging", "startup ideas", "weekend plans"],
        "platform": "telegram",
    },
    {
        "name": "Priya Nair",
        "pattern": "fading",          # was active, dropped off
        "days_since": 18,
        "total_msgs": 95,
        "topics": ["road trip", "music festivals", "college gossip"],
        "platform": "whatsapp",
    },
    {
        "name": "Jake Morris",
        "pattern": "stable",
        "days_since": 3,
        "total_msgs": 220,
        "topics": ["gym sessions", "NBA", "career advice"],
        "platform": "telegram",
    },
    {
        "name": "Riya Shah",
        "pattern": "ghosted",         # sent last msg, no reply >30 days
        "days_since": 41,
        "total_msgs": 48,
        "topics": ["birthday plans", "old school memories"],
        "platform": "whatsapp",
    },
    {
        "name": "Kabir Rao",
        "pattern": "active",
        "days_since": 0,
        "total_msgs": 510,
        "topics": ["study grind", "exam prep", "memes", "food spots"],
        "platform": "whatsapp",
    },
    {
        "name": "Zoe Lin",
        "pattern": "fading",
        "days_since": 27,
        "total_msgs": 72,
        "topics": ["coffee catch-up", "design portfolio", "freelance gigs"],
        "platform": "telegram",
    },
    {
        "name": "Sam Patel",
        "pattern": "close",           # high frequency, high sentiment
        "days_since": 1,
        "total_msgs": 650,
        "topics": ["everything", "life venting", "inside jokes", "travel plans"],
        "platform": "whatsapp",
    },
    {
        "name": "Maya Iyer",
        "pattern": "new",             # recently started talking
        "days_since": 2,
        "total_msgs": 34,
        "topics": ["internship advice", "networking", "LinkedIn"],
        "platform": "telegram",
    },
]

SAMPLE_MSGS = {
    "active":  ["hey what's up", "did you see that?", "lol same", "bro no way", "let's call later",
                "check this out", "you free tomorrow?", "that was insane", "agreed 100%", "hahaha"],
    "fading":  ["hope you're good!", "we should catch up soon", "miss hanging out tbh",
                "seen this?", "btw about that thing we discussed", "you okay?"],
    "ghosted": ["hey?", "you around?", "ping me when free", "all good?"],
    "stable":  ["solid", "makes sense", "yeah definitely", "for sure", "noted", "thanks man",
                "will do", "sounds good", "next week works"],
    "close":   ["bro I can't stop laughing", "no way this actually happened",
                "you're literally the only one who gets it", "STOP IT", "dying rn 💀",
                "okay okay hear me out", "update: worse", "tell me everything"],
    "new":     ["hey! so nice to meet you at the event", "definitely would love to connect",
                "that's really helpful thanks", "I'll keep that in mind", "sounds like a plan!"],
}


def _generate_messages_for_archetype(archetype: dict, total: int) -> List[Dict[str, Any]]:
    """Generate synthetic messages for a given contact archetype."""
    msgs = []
    now = datetime.now()
    pattern = archetype["pattern"]
    days_since = archetype["days_since"]

    # Work backwards from the last message date
    last_msg_time = now - timedelta(days=days_since, hours=random.randint(0, 12))

    # Distribute messages over time
    if pattern in ("active", "close"):
        span_days = 120
        daily_avg = max(1, total // span_days)
    elif pattern == "stable":
        span_days = 90
        daily_avg = max(1, total // span_days)
    elif pattern == "fading":
        span_days = 150    # was more active earlier
        daily_avg = 1
    elif pattern == "ghosted":
        span_days = 60
        daily_avg = 1
    else:
        span_days = 30
        daily_avg = 2

    pool = SAMPLE_MSGS.get(pattern, SAMPLE_MSGS["stable"])
    current_time = last_msg_time - timedelta(days=span_days)

    for _ in range(total):
        # Move time forward randomly
        gap_hours = max(0.1, random.expovariate(1.0 / (24 / max(daily_avg, 1))))
        current_time += timedelta(hours=gap_hours)
        if current_time > last_msg_time:
            current_time = last_msg_time - timedelta(minutes=random.randint(1, 60))

        sender = "user" if random.random() < 0.45 else archetype["name"]
        msgs.append({
            "timestamp": current_time,
            "sender": sender,
            "content": random.choice(pool) + (f" [{random.choice(archetype['topics'])}]" if random.random() < 0.1 else ""),
            "platform": archetype["platform"],
        })

    # Sort by timestamp
    msgs.sort(key=lambda x: x["timestamp"])

    # For ghosted: ensure last message is from user
    if pattern == "ghosted":
        msgs.append({
            "timestamp": last_msg_time,
            "sender": "user",
            "content": "hey, you around?",
            "platform": archetype["platform"],
        })

    return msgs


def generate_synthetic_dataset() -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate full synthetic dataset.
    Returns: { contact_name: [messages] }
    """
    dataset = {}
    for archetype in ARCHETYPES:
        msgs = _generate_messages_for_archetype(archetype, archetype["total_msgs"])
        dataset[archetype["name"]] = msgs
        print(f"  ✓ Generated {len(msgs)} messages for {archetype['name']} [{archetype['pattern']}]")
    return dataset


def export_as_csv(dataset: Dict[str, List[Dict]]) -> str:
    """Export synthetic dataset as CSV string."""
    import csv
    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["timestamp", "sender", "recipient", "message_text", "platform"])
    writer.writeheader()
    for contact, msgs in dataset.items():
        for msg in msgs:
            writer.writerow({
                "timestamp": msg["timestamp"].isoformat(),
                "sender": msg["sender"],
                "recipient": contact if msg["sender"] == "user" else "user",
                "message_text": msg["content"],
                "platform": msg["platform"],
            })
    return output.getvalue()
