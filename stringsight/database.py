from __future__ import annotations

from pathlib import Path
from typing import Generator
import threading

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from stringsight.config import settings

def _create_engine(database_url: str) -> Engine:
    """Create the SQLAlchemy engine.

    Args:
        database_url: SQLAlchemy database URL (e.g., `sqlite:///...` or `postgresql://...`).

    Returns:
        A configured SQLAlchemy Engine.
    """
    if database_url.startswith("sqlite:///"):
        sqlite_path = database_url.replace("sqlite:///", "", 1)
        Path(sqlite_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
        )

    return create_engine(database_url)


# Create SQLAlchemy engine
engine = _create_engine(settings.DATABASE_URL)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

_sqlite_init_lock = threading.Lock()
_sqlite_initialized = False


def init_db() -> None:
    """Initialize the database for local (SQLite) installs.

    For easy installs we default to a local SQLite DB; in that case, create tables
    automatically if they don't exist.
    """
    global _sqlite_initialized
    if not str(engine.url).startswith("sqlite"):
        return

    if _sqlite_initialized:
        return

    with _sqlite_init_lock:
        if _sqlite_initialized:
            return

    # Import models so they register with Base.metadata
    from stringsight.models import job  # noqa: F401
    from stringsight.models import user  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _sqlite_initialized = True


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session.

    Yields:
        A SQLAlchemy Session bound to the configured engine.
    """
    # In `stringsight launch`, the API app may be mounted under a parent FastAPI app,
    # in which case its startup hooks may not run. Ensure local SQLite tables exist
    # before the first DB use.
    init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
