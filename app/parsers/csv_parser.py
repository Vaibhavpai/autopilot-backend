"""
Parses generic CSV interaction logs.
Expected columns: timestamp, sender, recipient, message_text, platform
"""
import pandas as pd
from io import StringIO
from datetime import datetime
from typing import Dict, List, Any


REQUIRED_COLS = {"timestamp", "sender", "message_text"}
OPTIONAL_COLS = {"recipient", "platform"}


def parse_csv(content: str, your_name: str = "user") -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse CSV chat log. Returns { contact_name: [messages] }
    Flexible: tolerates missing optional columns.
    """
    df = pd.read_csv(StringIO(content))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    platform = "csv"
    contacts: Dict[str, List[Dict]] = {}

    for _, row in df.iterrows():
        sender = str(row["sender"]).strip()
        text = str(row.get("message_text", "")).strip()
        if not text or text == "nan":
            continue

        plat = str(row.get("platform", platform)).strip() if "platform" in df.columns else platform
        is_user = sender.lower() in (your_name.lower(), "user", "you", "me")

        # Determine contact name
        if is_user:
            contact = str(row.get("recipient", "Unknown")).strip() if "recipient" in df.columns else "Unknown"
        else:
            contact = sender

        if contact.lower() in ("unknown", "nan", ""):
            continue

        msg = {
            "timestamp": row["timestamp"].to_pydatetime(),
            "sender": "user" if is_user else sender,
            "content": text,
            "platform": plat,
        }

        if contact not in contacts:
            contacts[contact] = []
        contacts[contact].append(msg)

    return contacts
