"""add extraction_status and job_id indexes

Revision ID: a4e7c2f19b3d
Revises: 0df17b9623bb
Create Date: 2026-07-30 17:10:00.000000

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a4e7c2f19b3d'
down_revision: str | Sequence[str] | None = '0df17b9623bb'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(op.f('ix_candidates_extraction_status'), 'candidates', ['extraction_status'], unique=False)
    op.create_index('ix_matches_job_id', 'matches', ['job_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_matches_job_id', table_name='matches')
    op.drop_index(op.f('ix_candidates_extraction_status'), table_name='candidates')
