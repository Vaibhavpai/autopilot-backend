from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class Message(BaseModel):
    timestamp: datetime
    sender: str          # "user" or contact name
    content: str
    platform: str        # whatsapp | telegram | csv
    contact_id: str


class WeeklyActivity(BaseModel):
    week: str            # e.g. "2024-W03"
    message_count: int


class ContactProfile(BaseModel):
    contact_id: str
    name: str
    handle: Optional[str] = ""
    platform: str
    avatar: str          # initials
    total_messages: int
    last_message_at: Optional[datetime]
    days_since_last_message: int

    # Scoring components
    recency_score: float       # 0–100
    frequency_score: float     # 0–100
    response_ratio: float      # 0–1 (how often they reply)
    sentiment_avg: float       # -1 to +1 (VADER compound)
    health_score: float        # 0–100 composite

    # Behavioral flags
    drift_detected: bool
    is_ghosted: bool           # >30 days silent, you sent last
    drift_severity: str        # none | mild | moderate | severe

    # Enrichment
    last_topic: Optional[str] = ""
    tag: str                   # ACTIVE | CLOSE | STABLE | FADING | GHOSTED
    trend: str                 # up | stable | down
    weekly_activity: List[WeeklyActivity] = []


class ActionSuggestion(BaseModel):
    action_id: str
    contact_id: str
    contact_name: str
    action_type: str           # RE-ENGAGE | FOLLOW-UP | CHECK-IN | BIRTHDAY
    urgency: str               # CRITICAL | HIGH | MEDIUM | LOW
    suggested_message: str
    reason: str
    created_at: datetime
    status: str = "pending"    # pending | sent | dismissed


class PipelineRun(BaseModel):
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    contacts_processed: int
    actions_generated: int
    status: str                # running | completed | failed
    error: Optional[str] = None


class IngestResponse(BaseModel):
    success: bool
    contacts_found: int
    messages_parsed: int
    message: str
