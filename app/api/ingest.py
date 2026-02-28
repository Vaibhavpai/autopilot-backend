from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from app.core.db_helpers import (
    save_messages, clear_messages, count_messages, count_contacts,
    get_all_messages
)
from app.parsers.whatsapp_parser import extract_contacts_from_whatsapp
from app.parsers.telegram_parser import parse_telegram
from app.parsers.csv_parser import parse_csv
from app.parsers.synthetic_generator import generate_synthetic_dataset, export_as_csv
from fastapi.responses import PlainTextResponse

router = APIRouter()


async def _load_contacts(parsed: dict, platform_label: str) -> tuple[int, int]:
    """Merge parsed contacts into the DB. Returns (contacts_added, total_messages)."""
    total_msgs = 0
    for name, msgs in parsed.items():
        cid = name
        await save_messages(cid, msgs)
        total_msgs += len(msgs)
    return len(parsed), total_msgs


@router.post("/synthetic")
async def load_synthetic_data():
    """
    Generate and load synthetic demo dataset (8 contacts, ~2000 messages).
    Call this first to get something working immediately.
    """
    print("[INGEST] Generating synthetic dataset...")
    await clear_messages()
    dataset = generate_synthetic_dataset()
    contacts_added, total_msgs = await _load_contacts(dataset, "synthetic")
    return {
        "success": True,
        "contacts_found": contacts_added,
        "messages_parsed": total_msgs,
        "message": f"Synthetic dataset loaded: {contacts_added} contacts, {total_msgs} messages.",
        "contacts": list(dataset.keys()),
    }


@router.get("/synthetic/csv")
def download_synthetic_csv():
    """Download synthetic dataset as CSV (for testing parsers)."""
    dataset = generate_synthetic_dataset()
    csv_content = export_as_csv(dataset)
    return PlainTextResponse(content=csv_content, media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=synthetic_data.csv"})


@router.post("/whatsapp")
async def upload_whatsapp(
    file: UploadFile = File(...),
    your_name: str = Form(default="You"),
    clear_existing: bool = Form(default=False),
):
    """Upload WhatsApp exported .txt chat log."""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="File must be a .txt WhatsApp export")

    content = (await file.read()).decode("utf-8", errors="ignore")
    if clear_existing:
        await clear_messages()

    try:
        parsed = extract_contacts_from_whatsapp(content, your_name)
        contacts_added, total_msgs = await _load_contacts(parsed, "whatsapp")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Parse error: {str(e)}")

    return {
        "success": True,
        "contacts_found": contacts_added,
        "messages_parsed": total_msgs,
        "message": f"WhatsApp log parsed: {contacts_added} contacts, {total_msgs} messages.",
    }


@router.post("/telegram")
async def upload_telegram(
    file: UploadFile = File(...),
    your_name: str = Form(default="You"),
    clear_existing: bool = Form(default=False),
):
    """Upload Telegram result.json export."""
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="File must be Telegram result.json")

    content = (await file.read()).decode("utf-8", errors="ignore")
    if clear_existing:
        await clear_messages()

    try:
        parsed = parse_telegram(content, your_name)
        contacts_added, total_msgs = await _load_contacts(parsed, "telegram")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Parse error: {str(e)}")

    return {
        "success": True,
        "contacts_found": contacts_added,
        "messages_parsed": total_msgs,
        "message": f"Telegram export parsed: {contacts_added} contacts, {total_msgs} messages.",
    }


@router.post("/csv")
async def upload_csv(
    file: UploadFile = File(...),
    your_name: str = Form(default="user"),
    clear_existing: bool = Form(default=False),
):
    """Upload generic CSV interaction log."""
    content = (await file.read()).decode("utf-8", errors="ignore")
    if clear_existing:
        await clear_messages()

    try:
        parsed = parse_csv(content, your_name)
        contacts_added, total_msgs = await _load_contacts(parsed, "csv")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Parse error: {str(e)}")

    return {
        "success": True,
        "contacts_found": contacts_added,
        "messages_parsed": total_msgs,
        "message": f"CSV parsed: {contacts_added} contacts, {total_msgs} messages.",
    }


@router.get("/status")
async def ingest_status():
    """See what's currently loaded in MongoDB."""
    messages_dict = await get_all_messages()
    return {
        "contacts_loaded": len(messages_dict),
        "contacts": [
            {"name": name, "message_count": len(msgs)}
            for name, msgs in messages_dict.items()
        ],
        "total_messages": sum(len(m) for m in messages_dict.values()),
    }


@router.delete("/clear")
async def clear_data():
    """Wipe all loaded data (useful for re-ingestion)."""
    await clear_messages()
    return {"success": True, "message": "All data cleared."}
