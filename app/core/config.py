from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ANTHROPIC_API_KEY: str = "your-key-here"
    N8N_WEBHOOK_URL: str = "http://localhost:5678/webhook/autopilot"
    N8N_REMINDER_WEBHOOK: str = "http://localhost:5678/webhook/reminders"
    APP_ENV: str = "development"

    class Config:
        env_file = ".env"

settings = Settings()
