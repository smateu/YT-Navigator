#!/usr/bin/env python3
"""
Verification Script for OpenAI Migration
=========================================

This script verifies that all components are correctly configured for OpenAI API usage.

Usage:
    python scripts/verify_openai_setup.py
"""

import os
import sys

import structlog
from dotenv import load_dotenv

logger = structlog.get_logger(__name__)

# Load environment variables
load_dotenv()


def check_environment_variables() -> bool:
    """Check if all required environment variables are set.

    Returns:
        bool: True if all required variables are present
    """
    logger.info("=== Checking Environment Variables ===")

    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for all services",
        "POSTGRES_HOST": "PostgreSQL host",
        "POSTGRES_DB": "PostgreSQL database name",
        "POSTGRES_USER": "PostgreSQL user",
        "POSTGRES_PASSWORD": "PostgreSQL password",
    }

    optional_vars = {
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        "OPENAI_EMBEDDING_DIMENSIONS": "1536",
        "OPENAI_CHAT_MODEL": "gpt-4o-mini",
        "OPENAI_POWERFUL_MODEL": "gpt-4o",
    }

    all_ok = True

    # Check required variables
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            logger.info(f"✓ {var}: Set", description=description)
        else:
            logger.error(f"✗ {var}: NOT SET", description=description)
            all_ok = False

    # Check optional variables (show defaults)
    logger.info("\nOptional variables (will use defaults if not set):")
    for var, default in optional_vars.items():
        value = os.getenv(var)
        if value:
            logger.info(f"✓ {var}: {value}")
        else:
            logger.info(f"○ {var}: Using default ({default})")

    return all_ok


def check_openai_api() -> bool:
    """Test OpenAI API connection.

    Returns:
        bool: True if API is accessible
    """
    logger.info("\n=== Testing OpenAI API Connection ===")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Test API with a simple call
        logger.info("Testing embeddings API...")
        response = client.embeddings.create(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            input="test",
        )

        logger.info(f"✓ Embeddings API working", dimensions=len(response.data[0].embedding))

        # Test chat API
        logger.info("Testing chat API...")
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=5,
        )

        logger.info(f"✓ Chat API working")

        return True

    except Exception as e:
        logger.error(f"✗ OpenAI API test failed", error=str(e))
        return False


def check_dependencies() -> bool:
    """Check if all required packages are installed.

    Returns:
        bool: True if all packages are available
    """
    logger.info("\n=== Checking Python Dependencies ===")

    required_packages = {
        "openai": "OpenAI API client",
        "yt_dlp": "YouTube audio downloader",
        "langchain_openai": "LangChain OpenAI integration",
        "langchain_postgres": "LangChain PostgreSQL integration",
    }

    removed_packages = {
        "torch": "PyTorch (no longer needed)",
        "sentence_transformers": "Sentence Transformers (replaced by OpenAI)",
        "langchain_groq": "LangChain Groq (replaced by OpenAI)",
    }

    all_ok = True

    # Check required packages
    for package, description in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {package}: Installed", description=description)
        except ImportError:
            logger.error(f"✗ {package}: NOT INSTALLED", description=description)
            all_ok = False

    # Check removed packages (should NOT be present)
    logger.info("\nPackages that should be removed:")
    for package, description in removed_packages.items():
        try:
            __import__(package)
            logger.warning(f"⚠ {package}: Still installed (not required)", description=description)
        except ImportError:
            logger.info(f"✓ {package}: Not present", description=description)

    return all_ok


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed (required for yt-dlp).

    Returns:
        bool: True if FFmpeg is available
    """
    logger.info("\n=== Checking FFmpeg ===")

    import shutil

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path:
        logger.info(f"✓ FFmpeg found at: {ffmpeg_path}")
        return True
    else:
        logger.warning("✗ FFmpeg not found (required for Whisper transcription)")
        logger.info("  Install with:")
        logger.info("    Ubuntu/Debian: sudo apt-get install ffmpeg")
        logger.info("    macOS: brew install ffmpeg")
        return False


def check_database() -> bool:
    """Check database connection and structure.

    Returns:
        bool: True if database is accessible
    """
    logger.info("\n=== Checking Database Connection ===")

    try:
        import asyncio
        import asyncpg

        async def test_db():
            try:
                conn = await asyncpg.connect(
                    host=os.getenv("POSTGRES_HOST"),
                    port=int(os.getenv("POSTGRES_PORT", 5432)),
                    database=os.getenv("POSTGRES_DB"),
                    user=os.getenv("POSTGRES_USER"),
                    password=os.getenv("POSTGRES_PASSWORD"),
                )

                # Check if pgvector extension exists
                result = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )

                if result:
                    logger.info("✓ Database connected, pgvector extension available")
                else:
                    logger.warning("⚠ Database connected, but pgvector extension not found")

                await conn.close()
                return True

            except Exception as e:
                logger.error(f"✗ Database connection failed", error=str(e))
                return False

        return asyncio.run(test_db())

    except Exception as e:
        logger.error(f"✗ Database check failed", error=str(e))
        return False


def main():
    """Run all verification checks."""
    logger.info("=" * 60)
    logger.info("OpenAI Migration Verification")
    logger.info("=" * 60)

    results = {
        "Environment Variables": check_environment_variables(),
        "Python Dependencies": check_dependencies(),
        "FFmpeg": check_ffmpeg(),
        "Database Connection": check_database(),
        "OpenAI API": False,  # Will be checked last
    }

    # Check OpenAI API last (only if env vars are set)
    if results["Environment Variables"]:
        results["OpenAI API"] = check_openai_api()
    else:
        logger.warning("\n⚠ Skipping OpenAI API test (environment variables not set)")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Verification Summary")
    logger.info("=" * 60)

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {check}")

    all_passed = all(results.values())

    if all_passed:
        logger.info("\n✓ All checks passed! Your setup is ready to use OpenAI APIs.")
        logger.info("\nNext steps:")
        logger.info("1. Run database migration: python scripts/migrate_to_openai_embeddings.py")
        logger.info("2. Start the server: python manage.py runserver")
        logger.info("3. Re-scan channels to generate new embeddings")
    else:
        logger.error("\n✗ Some checks failed. Please fix the issues above.")
        logger.info("\nRefer to MIGRATION_GUIDE.md for detailed instructions.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
