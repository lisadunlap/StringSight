#!/usr/bin/env python3
"""Quick script to check recent jobs in the database."""

from stringsight.database import SessionLocal
from stringsight.db_models.job import Job
from datetime import datetime

db = SessionLocal()
try:
    # Get the 10 most recent jobs
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(10).all()

    print(f"\n{'='*100}")
    print(f"Recent Jobs (most recent first)")
    print(f"{'='*100}\n")

    if not jobs:
        print("No jobs found in database.")
    else:
        for job in jobs:
            print(f"Job ID: {job.id}")
            print(f"  Status: {job.status}")
            print(f"  Type: {job.job_type if hasattr(job, 'job_type') else 'N/A'}")
            print(f"  Progress: {job.progress * 100:.1f}%")
            print(f"  Result Path: {job.result_path}")
            print(f"  Created: {job.created_at}")
            if job.error_message:
                print(f"  Error: {job.error_message[:200]}")
            print(f"  {'-'*98}")

        print(f"\n{'='*100}")
        print("Jobs with Issues (failed, no result_path, or incomplete)")
        print(f"{'='*100}\n")

        problem_jobs = [j for j in jobs if j.status == 'failed' or
                       (j.status == 'completed' and not j.result_path)]

        if problem_jobs:
            for job in problem_jobs:
                print(f"⚠️  Job ID: {job.id}")
                print(f"   Status: {job.status}")
                print(f"   Result Path: {job.result_path or 'MISSING'}")
                if job.error_message:
                    print(f"   Error: {job.error_message}")
                print()
        else:
            print("✅ No problematic jobs found in recent 10")

finally:
    db.close()