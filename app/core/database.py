from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")

client = AsyncIOMotorClient(MONGO_URL)

db = client["autopilot"]

messages_collection = db["messages"]
contacts_collection = db["contacts"]
actions_collection = db["actions"]
pipeline_runs_collection = db["pipeline_runs"]