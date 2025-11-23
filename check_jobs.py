from stringsight.database import SessionLocal
from stringsight.models.job import Job
from datetime import datetime

db = SessionLocal()
jobs = db.query(Job).order_by(Job.created_at.desc()).limit(5).all()

print("\nRecent Jobs:")
print("-" * 100)
for job in jobs:
    print(f"ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Progress: {job.progress}")
    print(f"Error: {job.error_message}")
    print(f"Result: {job.result_path}")
    print(f"Created: {job.created_at}")
    print("-" * 100)

db.close()
