"""
Database Migration Script for OpenAI Embeddings
================================================

This script handles the migration from Sentence Transformers embeddings (384 dimensions)
to OpenAI embeddings (1536 or 3072 dimensions).

IMPORTANT: This migration requires re-processing all video transcripts with the new
embedding model. The old embeddings cannot be reused due to different dimensionality.

Usage:
    python scripts/migrate_to_openai_embeddings.py [--dry-run]

Options:
    --dry-run    Show what would be done without making changes

The script will:
1. Backup current embedding tables
2. Drop old embedding collections
3. Update PGVector extension for new dimensions
4. You'll need to re-scan channels to generate new embeddings
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

import asyncpg
import structlog
from django import setup
from dotenv import load_dotenv

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "yt_navigator.settings")
setup()

from django.conf import settings

logger = structlog.get_logger(__name__)

load_dotenv()


async def backup_embeddings_table(conn: asyncpg.Connection, dry_run: bool = False) -> bool:
    """Backup the current embeddings table.

    Args:
        conn: Database connection
        dry_run: If True, only show what would be done

    Returns:
        bool: True if successful
    """
    backup_table = f"langchain_pg_embedding_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        logger.info("Creating backup of embeddings table", backup_table=backup_table)

        if dry_run:
            logger.info("[DRY RUN] Would create backup table", backup_table=backup_table)
            return True

        await conn.execute(f"""
            CREATE TABLE {backup_table} AS
            SELECT * FROM langchain_pg_embedding;
        """)

        row_count = await conn.fetchval(f"SELECT COUNT(*) FROM {backup_table}")
        logger.info("Backup created successfully", backup_table=backup_table, rows=row_count)

        return True

    except Exception as e:
        logger.error("Failed to create backup", error=str(e))
        return False


async def drop_old_embeddings(conn: asyncpg.Connection, dry_run: bool = False) -> bool:
    """Drop old embedding collections.

    Args:
        conn: Database connection
        dry_run: If True, only show what would be done

    Returns:
        bool: True if successful
    """
    try:
        logger.info("Dropping old embeddings")

        if dry_run:
            logger.info("[DRY RUN] Would delete all rows from langchain_pg_embedding")
            return True

        # Get count before deletion
        count_before = await conn.fetchval("SELECT COUNT(*) FROM langchain_pg_embedding")
        logger.info("Current embeddings count", count=count_before)

        # Delete all embeddings (collections will be recreated with new dimensions)
        await conn.execute("DELETE FROM langchain_pg_embedding")

        logger.info("Old embeddings deleted successfully", deleted_count=count_before)
        return True

    except Exception as e:
        logger.error("Failed to drop old embeddings", error=str(e))
        return False


async def update_collection_metadata(conn: asyncpg.Connection, dry_run: bool = False) -> bool:
    """Update collection metadata for new embedding dimensions.

    Args:
        conn: Database connection
        dry_run: If True, only show what would be done

    Returns:
        bool: True if successful
    """
    try:
        logger.info("Updating collection metadata")

        if dry_run:
            logger.info("[DRY RUN] Would update langchain_pg_collection metadata")
            return True

        # Update the cmetadata for all collections to reflect new dimensions
        await conn.execute("""
            UPDATE langchain_pg_collection
            SET cmetadata = jsonb_set(
                COALESCE(cmetadata, '{}'::jsonb),
                '{embedding_dimensions}',
                $1::text::jsonb
            )
        """, str(settings.OPENAI_EMBEDDING_DIMENSIONS))

        logger.info("Collection metadata updated", new_dimensions=settings.OPENAI_EMBEDDING_DIMENSIONS)
        return True

    except Exception as e:
        logger.error("Failed to update collection metadata", error=str(e))
        return False


async def verify_migration(conn: asyncpg.Connection) -> dict:
    """Verify the migration was successful.

    Args:
        conn: Database connection

    Returns:
        dict: Migration verification results
    """
    try:
        # Check if embeddings table is empty
        embedding_count = await conn.fetchval("SELECT COUNT(*) FROM langchain_pg_embedding")

        # Check if backup tables exist
        backup_tables = await conn.fetch("""
            SELECT tablename FROM pg_tables
            WHERE tablename LIKE 'langchain_pg_embedding_backup_%'
            ORDER BY tablename DESC
        """)

        # Check collection metadata
        collections = await conn.fetch("""
            SELECT name, cmetadata
            FROM langchain_pg_collection
        """)

        results = {
            "embedding_count": embedding_count,
            "backup_tables": [row["tablename"] for row in backup_tables],
            "collections": [
                {
                    "name": row["name"],
                    "dimensions": row["cmetadata"].get("embedding_dimensions") if row["cmetadata"] else None
                }
                for row in collections
            ],
            "expected_dimensions": settings.OPENAI_EMBEDDING_DIMENSIONS,
        }

        return results

    except Exception as e:
        logger.error("Failed to verify migration", error=str(e))
        return {}


async def main(dry_run: bool = False):
    """Main migration function.

    Args:
        dry_run: If True, only show what would be done without making changes
    """
    logger.info("Starting migration to OpenAI embeddings", dry_run=dry_run)

    # Parse database URL
    db_config = {
        "host": os.getenv("POSTGRES_HOST"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }

    try:
        # Connect to database
        logger.info("Connecting to database", host=db_config["host"], database=db_config["database"])
        conn = await asyncpg.connect(**db_config)

        try:
            # Step 1: Backup current embeddings
            logger.info("=== Step 1: Backup Current Embeddings ===")
            if not await backup_embeddings_table(conn, dry_run):
                logger.error("Backup failed. Aborting migration.")
                return False

            # Step 2: Drop old embeddings
            logger.info("=== Step 2: Drop Old Embeddings ===")
            if not await drop_old_embeddings(conn, dry_run):
                logger.error("Failed to drop old embeddings. Aborting migration.")
                return False

            # Step 3: Update collection metadata
            logger.info("=== Step 3: Update Collection Metadata ===")
            if not await update_collection_metadata(conn, dry_run):
                logger.error("Failed to update collection metadata. Aborting migration.")
                return False

            # Step 4: Verify migration
            if not dry_run:
                logger.info("=== Step 4: Verify Migration ===")
                results = await verify_migration(conn)

                logger.info("Migration verification results:")
                logger.info(f"  - Embeddings count: {results.get('embedding_count', 'N/A')}")
                logger.info(f"  - Backup tables: {len(results.get('backup_tables', []))}")
                logger.info(f"  - Collections: {len(results.get('collections', []))}")
                logger.info(f"  - Expected dimensions: {results.get('expected_dimensions', 'N/A')}")

                if results.get('embedding_count', 0) > 0:
                    logger.warning("Embeddings table is not empty! This is unexpected.")

            logger.info("=== Migration Completed Successfully ===")
            logger.info("")
            logger.info("NEXT STEPS:")
            logger.info("1. Update your .env file with OPENAI_API_KEY")
            logger.info("2. Verify OPENAI_EMBEDDING_DIMENSIONS is set correctly")
            logger.info("3. Re-scan all channels to generate new embeddings:")
            logger.info("   - Go to each channel and click 'Scan Videos'")
            logger.info("   - This will generate embeddings using OpenAI API")
            logger.info("")

            return True

        finally:
            await conn.close()

    except Exception as e:
        logger.error("Migration failed", error=str(e))
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate from Sentence Transformers to OpenAI embeddings")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()

    success = asyncio.run(main(dry_run=args.dry_run))
    sys.exit(0 if success else 1)
