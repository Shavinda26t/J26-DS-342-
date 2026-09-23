import os

class Settings:
    PROJECT_NAME: str = "HDD Failure Prediction & Self-Healing Backend"
    API_V1_STR: str = "/api/v1"
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", 8000))

settings = Settings()
