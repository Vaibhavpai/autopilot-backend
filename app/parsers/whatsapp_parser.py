"""
Parses exported WhatsApp .txt chat logs.
Format: [DD/MM/YYYY, HH:MM:SS] Name: message
or:     DD/MM/YYYY, HH:MM - Name: message  (older export format)
"""
import re
from datetime import datetime
from typing import List, Dict, Any

# Covers both iOS and Android export formats
PATTERNS = [
    # [12/01/2024, 10:23:45] Alice: hey
    r"\[(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.*?):\s*(.*)",
    # 12/01/2024, 10:23 - Alice: hey
    r"(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(.*?):\s*(.*)",
]

DATE_FORMATS = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"]


def _try_parse_date(date_str: str, time_str: str) -> datetime:
    combined = f"{date_str} {time_str}"
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {combined}")


def parse_whatsapp(content: str, your_name: str = "You") -> List[Dict[str, Any]]:
    """
    Parse WhatsApp exported chat log.

    Args:
        content: raw text content of the exported file
        your_name: your display name in the chat (to mark sender as 'user')

    Returns:
        list of message dicts
    """
    messages = []
    lines = content.strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        matched = False
        for pattern in PATTERNS:
            m = re.match(pattern, line)
            if m:
                date_str, time_str, sender, text = m.groups()
                # Skip system messages
                if any(skip in text.lower() for skip in [
                    "messages and calls are end-to-end encrypted",
                    "changed the subject",
                    "added",
                    "left",
                    "<media omitted>",
                    "null",
                ]):
                    matched = True
                    break
                try:
                    ts = _try_parse_date(date_str, time_str)
                    messages.append({
                        "timestamp": ts,
                        "sender": "user" if sender.strip() == your_name else sender.strip(),
                        "content": text.strip(),
                        "platform": "whatsapp",
                    })
                    matched = True
                    break
                except ValueError:
                    continue

    return messages


def extract_contacts_from_whatsapp(content: str, your_name: str = "You") -> Dict[str, List[Dict]]:
    """Group parsed messages by contact name."""
    messages = parse_whatsapp(content, your_name)
    contacts: Dict[str, List[Dict]] = {}

    for msg in messages:
        if msg["sender"] == "user":
            continue
        name = msg["sender"]
        if name not in contacts:
            contacts[name] = []
        contacts[name].append(msg)

    return contacts
