from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ----------------------------
# MESSAGE
# ----------------------------

class Message(BaseModel):
    contact_id: str
    timestamp: datetime
    sender: str
    content: str
    platform: str  # whatsapp | telegram | csv

    # ML Outputs
    sentiment_score: Optional[float] = None
    importance_score: Optional[float] = None
    intent_label: Optional[str] = None
    attention_gap_flag: Optional[bool] = False
    plan_detected: Optional[bool] = False
    extracted_date: Optional[datetime] = None


# ----------------------------
# CONTACT
# ----------------------------

class Contact(BaseModel):
    contact_id: str
    name: Optional[str]
    platform: str
    avatar: Optional[str]

    health_score: Optional[float] = 0.0
    recency_score: Optional[float] = 0.0
    frequency_score: Optional[float] = 0.0
    response_ratio: Optional[float] = 0.0
    sentiment_avg: Optional[float] = 0.0

    drift_detected: Optional[bool] = False
    drift_severity: Optional[str] = None
    is_ghosted: Optional[bool] = False

    tag: Optional[str] = None
    trend: Optional[str] = None
    last_topic: Optional[str] = None
    days_since: Optional[int] = 0
    total_messages: Optional[int] = 0
    last_message_at: Optional[datetime] = None

    # ML Outputs
    churn_probability: Optional[float] = 0.0
    engagement_decay_rate: Optional[float] = 0.0
    delay_anomaly_score: Optional[float] = 0.0

    updated_at: Optional[datetime] = None



# ----------------------------
# ACTION
# ----------------------------

class Action(BaseModel):
    action_id: str
    contact_id: str
    contact_name: Optional[str]

    action_type: str
    urgency: str
    suggested_message: Optional[str]
    reason: Optional[str]
    status: str
    created_at: datetime

    # ML Confidence
    followup_probability: Optional[float] = None
    confidence_score: Optional[float] = None
    recommended_time: Optional[datetime] = None


# ----------------------------
# PIPELINE RUN
# ----------------------------

class PipelineRun(BaseModel):
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    contacts_processed: int
    actions_generated: int
    status: str
    trigger: str
    duration_seconds: Optional[float]
    error: Optional[str]