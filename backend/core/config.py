from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME:   str  = "CloudOS-RL"
    DEBUG:      bool = False
    API_HOST:   str  = "0.0.0.0"
    API_PORT:   int  = 8000

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
    ]

    KAFKA_BOOTSTRAP: str = "localhost:9092"
    AWS_REGION:      str = "us-east-1"

    MODEL_PATH:   str = "models/best/best_model"
    VECNORM_PATH: str = "models/vec_normalize.pkl"
    CONFIG_PATH:  str = "config/settings.yaml"
    LOG_LEVEL:    str = "INFO"

    class Config:
        env_file    = ".env"
        env_prefix  = "CLOUDOS_"