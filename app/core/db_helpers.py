"""
MongoDB helper functions for common operations
"""
from typing import Dict, List, Any, Optional
from bson import ObjectId
from app.core.database import (
    messages_collection,
    contacts_collection,
    actions_collection,
    pipeline_runs_collection
)


def _convert_objectid(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ObjectId to string and handle datetime serialization."""
    if doc is None:
        return None
    from datetime import datetime
    result = {}
    for key, value in doc.items():
        if key == "_id" and isinstance(value, ObjectId):
            continue  # Skip MongoDB _id field
        elif isinstance(value, ObjectId):
            result[key] = str(value)
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, dict):
            result[key] = _convert_objectid(value)
        elif isinstance(value, list):
            result[key] = [_convert_objectid(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    return result


# Messages helpers
async def get_all_messages() -> Dict[str, List[Dict[str, Any]]]:
    """Get all messages grouped by contact_id."""
    cursor = messages_collection.find({})
    messages_dict = {}
    async for doc in cursor:
        contact_id = doc.get("contact_id", "")
        if contact_id not in messages_dict:
            messages_dict[contact_id] = []
        messages_dict[contact_id].append({
            "timestamp": doc.get("timestamp"),
            "sender": doc.get("sender"),
            "content": doc.get("content"),
            "platform": doc.get("platform"),
        })
    return messages_dict


async def save_messages(contact_id: str, messages: List[Dict[str, Any]]):
    """Save messages for a contact."""
    from datetime import datetime
    docs = []
    for msg in messages:
        timestamp = msg["timestamp"]
        # Convert string timestamps to datetime if needed
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now()
        docs.append({
            "contact_id": contact_id,
            "timestamp": timestamp,
            "sender": msg["sender"],
            "content": msg["content"],
            "platform": msg.get("platform", "unknown"),
        })
    if docs:
        await messages_collection.insert_many(docs)


async def clear_messages():
    """Clear all messages."""
    await messages_collection.delete_many({})


async def count_messages() -> int:
    """Count total messages."""
    return await messages_collection.count_documents({})


async def count_contacts() -> int:
    """Count unique contacts."""
    return len(await messages_collection.distinct("contact_id"))


# Contacts helpers
async def save_contact(contact: Dict[str, Any]):
    """Save or update a contact profile."""
    await contacts_collection.update_one(
        {"contact_id": contact["contact_id"]},
        {"$set": contact},
        upsert=True
    )


async def get_all_contacts() -> List[Dict[str, Any]]:
    """Get all contact profiles."""
    cursor = contacts_collection.find({})
    return [_convert_objectid(doc) async for doc in cursor]


async def get_contact(contact_id: str) -> Optional[Dict[str, Any]]:
    """Get a single contact by ID."""
    doc = await contacts_collection.find_one({"contact_id": contact_id})
    return _convert_objectid(doc) if doc else None


async def clear_contacts():
    """Clear all contacts."""
    await contacts_collection.delete_many({})


# Actions helpers
async def save_actions(actions: List[Dict[str, Any]]):
    """Save multiple actions."""
    if actions:
        await actions_collection.insert_many(actions)


async def clear_actions():
    """Clear all actions."""
    await actions_collection.delete_many({})


async def get_all_actions() -> List[Dict[str, Any]]:
    """Get all actions."""
    cursor = actions_collection.find({})
    return [_convert_objectid(doc) async for doc in cursor]


async def update_action_status(action_id: str, status: str) -> bool:
    """Update action status. Returns True if updated."""
    result = await actions_collection.update_one(
        {"action_id": action_id},
        {"$set": {"status": status}}
    )
    return result.modified_count > 0


async def delete_action(action_id: str) -> bool:
    """Delete an action. Returns True if deleted."""
    result = await actions_collection.delete_one({"action_id": action_id})
    return result.deleted_count > 0


# Pipeline log helpers
async def save_pipeline_run(run: Dict[str, Any]):
    """Save a pipeline run log."""
    await pipeline_runs_collection.insert_one(run)


async def update_pipeline_run(run_id: str, updates: Dict[str, Any]):
    """Update a pipeline run."""
    await pipeline_runs_collection.update_one(
        {"run_id": run_id},
        {"$set": updates}
    )


async def get_last_pipeline_run() -> Optional[Dict[str, Any]]:
    """Get the most recent pipeline run."""
    cursor = pipeline_runs_collection.find().sort("started_at", -1).limit(1)
    async for doc in cursor:
        return _convert_objectid(doc)
    return None


async def get_pipeline_history() -> List[Dict[str, Any]]:
    """Get all pipeline runs, newest first."""
    cursor = pipeline_runs_collection.find().sort("started_at", -1)
    return [_convert_objectid(doc) async for doc in cursor]
