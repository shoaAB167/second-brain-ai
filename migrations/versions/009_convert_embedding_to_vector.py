"""Convert experiences.embedding from JSON to VECTOR(1536)

Revision ID: 009_convert_embedding_to_vector
Revises: 008_experiences_embedding
Create Date: 2026-08-29 16:35:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision: str = "009_convert_embedding_to_vector"
down_revision: Union[str, None] = "008_experiences_embedding"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        # 1. Ensure pgvector extension is enabled
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 2. Check and convert column from JSON to VECTOR(1536) safely
        op.execute("""
            DO $$
            BEGIN
                -- 1. Ensure user_id column is UUID type
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'experiences' 
                      AND column_name = 'user_id' 
                      AND data_type = 'character varying'
                ) THEN
                    ALTER TABLE experiences 
                        ALTER COLUMN user_id TYPE uuid 
                        USING (CASE WHEN user_id IS NULL OR user_id = '' THEN NULL ELSE user_id::uuid END);
                END IF;

                -- 2. Check if embedding column exists and is json
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'experiences' 
                      AND column_name = 'embedding' 
                      AND (data_type = 'json' OR data_type = 'jsonb')
                ) THEN
                    -- Validate dimensions of non-null embeddings (must be 1536)
                    IF EXISTS (
                        SELECT 1 FROM experiences 
                        WHERE embedding IS NOT NULL 
                          AND json_array_length(embedding) != 1536
                    ) THEN
                        RAISE EXCEPTION 'Cannot convert experiences.embedding to vector(1536): found embeddings with invalid dimensions.';
                    END IF;

                    -- Alter column type safely preserving data and NULLs
                    ALTER TABLE experiences 
                        ALTER COLUMN embedding TYPE vector(1536) 
                        USING (
                            CASE 
                                WHEN embedding IS NULL THEN NULL 
                                ELSE (embedding::text)::vector(1536) 
                            END
                        );
                END IF;
            END $$;
        """)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'experiences' 
                      AND column_name = 'embedding' 
                      AND udt_name = 'vector'
                ) THEN
                    ALTER TABLE experiences 
                        ALTER COLUMN embedding TYPE json 
                        USING (
                            CASE 
                                WHEN embedding IS NULL THEN NULL 
                                ELSE (embedding::text)::json 
                            END
                        );
                END IF;
            END $$;
        """)
