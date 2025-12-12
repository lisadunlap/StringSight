import uuid
from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text, Uuid
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from stringsight.database import Base

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Job type: extract, pipeline, cluster
    job_type = Column(String, default="extract", index=True)

    # Status: queued, running, completed, failed, cancelled
    status = Column(String, default="queued", index=True)
    progress = Column(Float, default=0.0)
    
    # Path to results in storage (e.g., s3://bucket/user/job/results.jsonl)
    result_path = Column(String, nullable=True)
    
    # Error message if failed
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", backref="jobs")
