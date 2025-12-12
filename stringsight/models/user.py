import uuid
from sqlalchemy import Boolean, Column, DateTime, String, Uuid
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from stringsight.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships are defined in the other models using backref or explicitly here
    # Job model defines: user = relationship("User", backref="jobs")







