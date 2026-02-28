from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import contacts, pipeline, actions, ingest
from app.core.scheduler import start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(
    title="Autopilot Social — Backend API",
    description="AI-driven relationship intelligence pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(contacts.router, prefix="/api/contacts", tags=["Contacts"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(actions.router,  prefix="/api/actions",  tags=["Actions"])
app.include_router(ingest.router,   prefix="/api/ingest",   tags=["Ingest"])

@app.get("/")
def root():
    return {"status": "Autopilot Social API running", "version": "0.1.0"}

@app.get("/health")
def health():
    return {"status": "ok"}