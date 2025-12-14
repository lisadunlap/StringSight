from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


def _default_database_url() -> str:
    """Return the default database URL for local installs.

    Returns:
        A SQLAlchemy database URL string. Defaults to a SQLite file under
        `~/.stringsight/stringsight.db` so `stringsight launch` works without
        requiring a running Postgres instance.
    """
    base_dir = Path.home() / ".stringsight"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "stringsight.db"
    return f"sqlite:///{db_path}"

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = Field(default_factory=_default_database_url)
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Auth
    SECRET_KEY: str = "development_secret_key_change_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Storage
    STORAGE_TYPE: str = "local"  # local, s3
    S3_BUCKET: str = "stringsight-results"
    S3_ENDPOINT_URL: Optional[str] = "http://localhost:9000"
    AWS_ACCESS_KEY_ID: Optional[str] = "minioadmin"
    AWS_SECRET_ACCESS_KEY: Optional[str] = "minioadmin"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    json_logs: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

settings = Settings()
