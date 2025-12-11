# OpenAI Migration - Quick Reference

## What Changed?

YT-Navigator has been migrated from GPU-dependent local models to OpenAI APIs for server-friendly deployment.

## Quick Start

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Configure Environment

```bash
# Update .env file
OPENAI_API_KEY=sk-your-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_POWERFUL_MODEL=gpt-4o
```

### 3. Verify Setup

```bash
python scripts/verify_openai_setup.py
```

### 4. Migrate Database

```bash
# Dry run first
python scripts/migrate_to_openai_embeddings.py --dry-run

# Run migration
python scripts/migrate_to_openai_embeddings.py
```

### 5. Re-scan Channels

After migration, re-scan all channels through the web interface to generate new embeddings.

## Key Changes

### Files Created

- `app/services/scraping/whisper_transcript.py` - OpenAI Whisper transcription
- `app/services/chunks_reranker/openai_reranker.py` - GPT-based reranking
- `scripts/migrate_to_openai_embeddings.py` - Database migration script
- `scripts/verify_openai_setup.py` - Setup verification
- `MIGRATION_GUIDE.md` - Complete migration documentation

### Files Modified

- `pyproject.toml` - Updated dependencies (removed torch, sentence-transformers; added openai, yt-dlp)
- `.env.example` - Added OpenAI configuration
- `yt_navigator/settings.py` - Replaced model settings with OpenAI config
- `app/services/scraping/transcript.py` - Added Whisper fallback
- `app/services/vector_database/base.py` - Replaced HuggingFace with OpenAI embeddings
- `app/services/chunks_reranker/reranker.py` - Wrapper for OpenAI reranker
- `app/services/agent/main_graph.py` - Replaced Groq with OpenAI
- `app/services/agent/react_graph.py` - Replaced Groq with OpenAI

## Architecture Changes

### Before

```
YouTube Video → youtube-transcript-api → Sentence Transformers (GPU) → PGVector
                                         ↓
                                    Cross-encoder (GPU) → Groq LLMs
```

### After

```
YouTube Video → youtube-transcript-api (fallback: Whisper API) → OpenAI Embeddings → PGVector
                                                                   ↓
                                                              GPT-4o-mini → OpenAI Chat
```

## Cost Estimate

Typical usage for 50 videos:

- Transcription (if needed): ~$3.00
- Embeddings: ~$0.01
- Chat (100 queries): ~$0.03
- **Total: ~$3.04**

Much lower than GPU infrastructure costs!

## Benefits

✅ No GPU required - runs on any server
✅ Better transcription - works with any video
✅ More powerful models - GPT-4o available
✅ Predictable costs - pay per use
✅ Easier deployment - no ML infrastructure needed

## Need Help?

1. Check `MIGRATION_GUIDE.md` for detailed instructions
2. Run `python scripts/verify_openai_setup.py` to diagnose issues
3. Review logs in `logs/` directory
4. Check OpenAI API status: https://status.openai.com/

## Rollback

If needed, see "Rollback Plan" section in `MIGRATION_GUIDE.md`.
