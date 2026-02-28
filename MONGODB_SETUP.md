# MongoDB Setup Guide

## Installation

1. **Install MongoDB** (if not already installed):
   ```bash
   # Windows (using Chocolatey)
   choco install mongodb
   
   # Or download from: https://www.mongodb.com/try/download/community
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start MongoDB**:
   ```bash
   # Windows
   net start MongoDB
   
   # Or run manually:
   mongod --dbpath C:\data\db
   ```

## Configuration

The app uses MongoDB with the following default connection:
- **URL**: `mongodb://localhost:27017` (default)
- **Database**: `autopilot`
- **Collections**:
  - `messages` - Raw parsed messages
  - `contacts` - Computed contact profiles
  - `actions` - AI-generated action suggestions
  - `pipeline_runs` - Pipeline execution history

## Environment Variables

You can override the MongoDB URL using:
```bash
export MONGO_URL="mongodb://localhost:27017"
# Or for MongoDB Atlas:
export MONGO_URL="mongodb+srv://user:pass@cluster.mongodb.net/autopilot"
```

## Database Structure

### Messages Collection
```json
{
  "contact_id": "contact_name",
  "timestamp": ISODate("2024-01-01T12:00:00Z"),
  "sender": "user" | "contact_name",
  "content": "message text",
  "platform": "whatsapp" | "telegram" | "csv"
}
```

### Contacts Collection
```json
{
  "contact_id": "contact_name",
  "name": "Contact Name",
  "health_score": 75.5,
  "tag": "ACTIVE",
  ...
}
```

### Actions Collection
```json
{
  "action_id": "uuid",
  "contact_id": "contact_name",
  "urgency": "CRITICAL",
  "status": "pending",
  ...
}
```

### Pipeline Runs Collection
```json
{
  "run_id": "abc12345",
  "started_at": ISODate("..."),
  "completed_at": ISODate("..."),
  "status": "completed",
  ...
}
```

## Indexes

Indexes are automatically created on startup:
- `messages.contact_id`
- `messages.timestamp`
- `contacts.contact_id` (unique)
- `actions.contact_id`
- `actions.status`
- `pipeline_runs.run_id`

## Troubleshooting

### Connection Issues
- Make sure MongoDB is running: `mongod --version`
- Check if port 27017 is accessible
- Verify connection string in environment variables

### Data Not Persisting
- Check MongoDB logs for errors
- Verify write permissions
- Ensure collections are being created

### Performance Issues
- Indexes are created automatically
- Consider adding compound indexes for complex queries
- Monitor MongoDB performance metrics
