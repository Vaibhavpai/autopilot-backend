# 🤖 Autopilot Social — Backend

AI-driven relationship intelligence pipeline built with FastAPI.

## Stack
- **FastAPI** — REST API
- **VADER** — Sentiment analysis  
- **scikit-learn** — Drift detection
- **Claude API** — Action generation
- **n8n** — Automation (schedules, emails)
- **APScheduler** — In-process scheduling

---

## Quick Start

```bash
# 1. Clone & install
cd autopilot-backend
pip install -r requirements.txt

# 2. Set env vars
cp .env.example .env
# → Add your ANTHROPIC_API_KEY

# 3. Run the server
uvicorn app.main:app --reload --port 8000

# 4. Open API docs
open http://localhost:8000/docs
```

---

## First Run (3 API calls)

```bash
# Step 1: Load synthetic demo data
curl -X POST http://localhost:8000/api/ingest/synthetic

# Step 2: Run the full pipeline
curl -X POST http://localhost:8000/api/pipeline/run/sync

# Step 3: Get results
curl http://localhost:8000/api/contacts/summary
curl http://localhost:8000/api/actions/
```

---

## API Reference

### Ingest
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ingest/synthetic` | Load demo data instantly |
| POST | `/api/ingest/whatsapp` | Upload WhatsApp .txt export |
| POST | `/api/ingest/telegram` | Upload Telegram result.json |
| POST | `/api/ingest/csv` | Upload generic CSV log |
| GET  | `/api/ingest/status` | See what's loaded |
| DELETE | `/api/ingest/clear` | Wipe all data |

### Pipeline
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/pipeline/run` | Async pipeline run (recommended) |
| POST | `/api/pipeline/run/sync` | Sync run (blocks, for testing) |
| GET  | `/api/pipeline/status` | Last run status |
| GET  | `/api/pipeline/history` | All run logs |

### Contacts
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/contacts/` | All contacts (filterable) |
| GET  | `/api/contacts/summary` | Dashboard stats |
| GET  | `/api/contacts/{id}` | Single contact profile |

### Actions
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/actions/` | All AI actions |
| PATCH | `/api/actions/{id}/status` | Mark sent/dismissed |
| DELETE | `/api/actions/{id}` | Remove action |

---

## Scoring Formula

```
Health Score = 
  0.30 × Recency Score     (exponential decay from last message)
  0.30 × Frequency Score   (msgs/week over last 30d, normalized)
  0.20 × Response Ratio    (% of your msgs they reply to)
  0.20 × Sentiment Score   (VADER compound, normalized 0-100)
```

## Tags
| Tag | Condition |
|-----|-----------|
| CLOSE | Score ≥ 80, high frequency |
| ACTIVE | Score ≥ 60, no drift |
| STABLE | Score 40–60 |
| FADING | Drift detected, score < 50 |
| GHOSTED | User sent last msg, >30 days silent |

---

## n8n Setup

1. Install n8n: `npx n8n`
2. Go to `http://localhost:5678`
3. Import `n8n/autopilot_workflow.json`
4. Configure your email credentials in the Email node
5. Activate the workflow

**Webhooks n8n listens on:**
- `POST /webhook/autopilot` — pipeline complete event
- `POST /webhook/reminders` — new critical actions

**Schedule:** Pipeline auto-runs every 6h via n8n trigger

---

## Project Structure

```
autopilot-backend/
├── app/
│   ├── main.py                    # FastAPI app
│   ├── api/
│   │   ├── contacts.py            # GET contacts, summary
│   │   ├── pipeline.py            # Trigger pipeline runs
│   │   ├── actions.py             # AI action suggestions
│   │   └── ingest.py              # File upload endpoints
│   ├── core/
│   │   ├── config.py              # Settings / env vars
│   │   ├── database.py            # In-memory store
│   │   └── scheduler.py           # APScheduler setup
│   ├── models/
│   │   └── schemas.py             # Pydantic models
│   ├── parsers/
│   │   ├── whatsapp_parser.py     # .txt log parser
│   │   ├── telegram_parser.py     # result.json parser
│   │   ├── csv_parser.py          # generic CSV parser
│   │   └── synthetic_generator.py # demo data generator
│   └── services/
│       ├── scoring_engine.py      # ⭐ Scoring + drift detection
│       ├── action_generator.py    # ⭐ Claude AI actions
│       ├── pipeline.py            # ⭐ Orchestrator
│       └── n8n_client.py          # Webhook client
└── n8n/
    └── autopilot_workflow.json    # Import into n8n
```
