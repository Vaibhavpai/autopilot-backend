"""
Simple in-memory store — swap for SQLAlchemy + Postgres in production.
All data lives in these dicts keyed by contact_id.
"""
from typing import Dict, List, Any

# Raw parsed messages: { contact_id: [{ timestamp, sender, content, platform }] }
messages_db: Dict[str, List[Dict[str, Any]]] = {}

# Computed contact profiles: { contact_id: ContactProfile }
contacts_db: Dict[str, Any] = {}

# AI-generated action suggestions
actions_db: List[Dict[str, Any]] = []

# Pipeline run log
pipeline_log: List[Dict[str, Any]] = []
