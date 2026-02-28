"""
Parses Telegram exported JSON (result.json from Telegram Desktop export).
"""
import json
from datetime import datetime
from typing import List, Dict, Any


def parse_telegram(content: str, your_name: str = "You") -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse Telegram result.json export.
    Returns dict: { contact_name: [messages] }
    """
    data = json.loads(content)
    contacts: Dict[str, List[Dict]] = {}

    # Telegram export can have 'chats' or be a single chat object
    chats = []
    if "chats" in data:
        chats = data["chats"].get("list", [])
    elif "messages" in data:
        # Single chat export
        chats = [data]

    for chat in chats:
        chat_type = chat.get("type", "")
        if chat_type not in ("personal_chat", "private_group", ""):
            continue

        contact_name = chat.get("name", "Unknown")
        if contact_name == your_name:
            continue

        messages = []
        for msg in chat.get("messages", []):
            if msg.get("type") != "message":
                continue

            sender = msg.get("from", "")
            # Text can be string or list of entities
            raw_text = msg.get("text", "")
            if isinstance(raw_text, list):
                text = "".join(
                    part if isinstance(part, str) else part.get("text", "")
                    for part in raw_text
                )
            else:
                text = raw_text

            if not text.strip():
                continue

            try:
                ts = datetime.fromisoformat(msg["date"])
            except (KeyError, ValueError):
                continue

            messages.append({
                "timestamp": ts,
                "sender": "user" if sender == your_name else sender,
                "content": text.strip(),
                "platform": "telegram",
            })

        if messages:
            if contact_name not in contacts:
                contacts[contact_name] = []
            contacts[contact_name].extend(messages)

    return contacts
