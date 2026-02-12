import os
from typing import List
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # Model Settings
    MODEL_PATH: str = "../models/fast_model.pth"
    MODEL_INPUT_SIZE: tuple = (224, 224)

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # React development server
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]

    # File Upload Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    SAVE_UPLOADED_IMAGES: bool = False
    UPLOAD_DIR: str = "uploads"
    MAX_BATCH_SIZE: int = 10

    # Security (optional)
    API_KEY: str = ""
    RATE_LIMIT: str = "100/minute"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Create necessary directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)