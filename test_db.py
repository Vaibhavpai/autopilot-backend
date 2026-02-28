import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
from app.core.database import client, db

async def test_connection():
    try:
        url = os.getenv("MONGO_URL")
        print(f"Testing connection to: {url}")
        
        # Ping the database
        await db.command("ping")
        print("Successfully connected to MongoDB!")
        
        # List collections
        collections = await db.list_collection_names()
        print(f"Available collections in 'autopilot' db: {collections}")
        
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(test_connection())
