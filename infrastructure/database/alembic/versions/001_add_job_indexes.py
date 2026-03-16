"""add indexes to jobs table

Revision ID: 001_add_job_indexes
Revises: 85094035cce5
Create Date: 2024-01-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_add_job_indexes'
down_revision: Union[str, Sequence[str], None] = '85094035cce5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add performance indexes to jobs table."""
    # Add index on created_at for chronological queries
    op.create_index('ix_jobs_created_at', 'jobs', ['created_at'])

    # Add compound index on (status, created_at) for filtering active jobs by date
    op.create_index('ix_jobs_status_created_at', 'jobs', ['status', 'created_at'])


def downgrade() -> None:
    """Remove performance indexes from jobs table."""
    op.drop_index('ix_jobs_status_created_at', table_name='jobs')
    op.drop_index('ix_jobs_created_at', table_name='jobs')
