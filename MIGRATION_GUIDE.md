# Migration Guide: From GPU Models to OpenAI APIs

This guide explains how to migrate YT-Navigator from GPU-dependent local models to OpenAI APIs.

## Overview of Changes

### What's Being Replaced

| Component         | Old (GPU-dependent)                                 | New (OpenAI API)                                 |
| ----------------- | --------------------------------------------------- | ------------------------------------------------ |
| **Transcription** | youtube-transcript-api only                         | youtube-transcript-api + OpenAI Whisper fallback |
| **Embeddings**    | Sentence Transformers (bge-small-en-v1.5, 384 dims) | OpenAI text-embedding-3-small (1536 dims)        |
| **Reranking**     | Cross-encoder (ms-marco-MiniLM-L-6-v2)              | GPT-4o-mini relevance scoring                    |
| **Chat Agent**    | Groq (llama-3.1-8b-instant, qwen-qwq-32b)           | OpenAI (gpt-4o-mini, gpt-4o)                     |
| **Vector DB**     | PGVector (kept, but dimensions change)              | PGVector (with new dimensions)                   |

## Prerequisites

1. **OpenAI API Key**: Get your API key from https://platform.openai.com/
2. **Python 3.13+**: Ensure your Python version is compatible
3. **Database Backup**: Always backup your database before migration
4. **FFmpeg**: Required by yt-dlp for audio extraction (if not already installed)

```bash
# Install FFmpeg (Ubuntu/Debian)
sudo apt-get install ffmpeg

# Install FFmpeg (macOS)
brew install ffmpeg
```

## Migration Steps

### Step 1: Update Dependencies

The dependencies have already been updated in `pyproject.toml`. Install them:

```bash
pip install -e .
```

**New dependencies added:**

- `openai>=1.54.0` - OpenAI API client
- `yt-dlp>=2024.12.6` - YouTube audio downloader
- `langchain-openai>=0.2.0` - LangChain OpenAI integration

**Dependencies removed:**

- `torch` - No longer needed (no GPU processing)
- `sentence-transformers` - Replaced by OpenAI embeddings
- `langchain-groq` - Replaced by langchain-openai
- `langchain-huggingface` - Replaced by OpenAI embeddings

### Step 2: Update Environment Variables

Update your `.env` file with OpenAI configuration:

```bash
# Replace GROQ_API_KEY with OPENAI_API_KEY
OPENAI_API_KEY=sk-your-openai-api-key-here

# Add OpenAI model configuration (optional, uses defaults if not set)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_POWERFUL_MODEL=gpt-4o

# Batch sizes for API calls (optional)
EMBEDDING_BATCH_SIZE=100
RERANKING_BATCH_SIZE=20
```

### Step 3: Database Migration

**⚠️ IMPORTANT: This step requires re-scanning all channels!**

The embedding dimensions have changed from 384 to 1536, which means all existing embeddings are incompatible and must be regenerated.

#### Option A: Automated Migration Script (Recommended)

```bash
# Dry run first to see what will happen
python scripts/migrate_to_openai_embeddings.py --dry-run

# Run the actual migration
python scripts/migrate_to_openai_embeddings.py
```

The script will:

1. Backup your current embeddings to a timestamped table
2. Clear all embeddings from the database
3. Update collection metadata for new dimensions
4. Verify the migration was successful

#### Option B: Manual Migration

```sql
-- 1. Backup current embeddings
CREATE TABLE langchain_pg_embedding_backup AS
SELECT * FROM langchain_pg_embedding;

-- 2. Clear old embeddings
DELETE FROM langchain_pg_embedding;

-- 3. Update collection metadata (if needed)
UPDATE langchain_pg_collection
SET cmetadata = jsonb_set(
    COALESCE(cmetadata, '{}'::jsonb),
    '{embedding_dimensions}',
    '1536'::text::jsonb
);
```

### Step 4: Django Migrations

Run Django migrations to ensure database schema is up to date:

```bash
python manage.py migrate
```

### Step 5: Re-scan All Channels

After migration, you need to re-scan all channels to generate new embeddings:

1. Log into the application
2. For each channel:
   - Navigate to the channel page
   - Click "Scan Videos"
   - Select number of videos to scan
   - Wait for processing to complete

**Note**: This will now use OpenAI Whisper API for videos without transcripts, and OpenAI embeddings for all text segments.

### Step 6: Verify the Migration

Test the application functionality:

1. **Search**: Try searching for content across videos
2. **Chat**: Test the chatbot with various queries
3. **Transcripts**: Scan a video without subtitles to verify Whisper integration

## Cost Estimation

### OpenAI API Pricing (as of Dec 2024)

| Service        | Model                  | Cost                                                |
| -------------- | ---------------------- | --------------------------------------------------- |
| **Whisper**    | whisper-1              | $0.006 / minute of audio                            |
| **Embeddings** | text-embedding-3-small | $0.02 / 1M tokens (~3,000 pages)                    |
| **Chat**       | gpt-4o-mini            | $0.150 / 1M input tokens, $0.600 / 1M output tokens |
| **Chat**       | gpt-4o                 | $2.50 / 1M input tokens, $10.00 / 1M output tokens  |

### Example Cost Calculation

**Scenario**: Processing a YouTube channel with 50 videos

1. **Transcription** (if videos lack subtitles):

   - Average video: 10 minutes
   - 50 videos = 500 minutes
   - Cost: 500 × $0.006 = **$3.00**

2. **Embeddings**:

   - ~100 chunks per video
   - 5,000 chunks × 50 tokens avg = 250,000 tokens
   - Cost: (250K / 1M) × $0.02 = **$0.005**

3. **Chat interactions** (100 queries):
   - Average: 1,000 input tokens, 200 output tokens per query
   - Input: (100K / 1M) × $0.15 = $0.015
   - Output: (20K / 1M) × $0.60 = $0.012
   - Cost: **$0.027**

**Total estimated cost**: ~$3.03 for 50 videos + 100 chat queries

### Cost Optimization Tips

1. **Use youtube-transcript-api first**: Whisper is only used as fallback
2. **Batch embeddings**: Process multiple chunks together (already configured)
3. **Use gpt-4o-mini**: More cost-effective for most tasks
4. **Set limits**: Configure max videos per scan to control costs
5. **Cache responses**: The vector DB stores results, so repeated queries are free

## Rollback Plan

If you need to rollback to the old system:

### 1. Restore Dependencies

```bash
# In pyproject.toml, restore:
# - torch==2.6.0
# - sentence-transformers==3.4.1
# - langchain-groq==0.2.5
# - langchain-huggingface

pip install -e .
```

### 2. Restore Database

```sql
-- Restore from backup (replace with your backup table name)
DELETE FROM langchain_pg_embedding;

INSERT INTO langchain_pg_embedding
SELECT * FROM langchain_pg_embedding_backup_YYYYMMDD_HHMMSS;
```

### 3. Restore Environment Variables

```bash
# In .env
GROQ_API_KEY=your-groq-key
# Remove OPENAI_* variables
```

### 4. Restore Code Changes

```bash
# Revert the code changes
git checkout <previous-commit>
```

## Troubleshooting

### Issue: "OpenAI API key not found"

**Solution**: Ensure `OPENAI_API_KEY` is set in your `.env` file and the application has been restarted.

### Issue: "Dimension mismatch in vector store"

**Solution**: Run the migration script to clear old embeddings:

```bash
python scripts/migrate_to_openai_embeddings.py
```

### Issue: "FFmpeg not found" when using Whisper

**Solution**: Install FFmpeg:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Issue: "Rate limit exceeded" from OpenAI

**Solution**:

1. Check your OpenAI account tier and limits
2. Reduce `EMBEDDING_BATCH_SIZE` in .env
3. Add delays between API calls if needed
4. Upgrade your OpenAI account tier

### Issue: High OpenAI API costs

**Solutions**:

1. Use `text-embedding-3-small` instead of `text-embedding-3-large`
2. Keep `gpt-4o-mini` for most tasks, only use `gpt-4o` when needed
3. Ensure youtube-transcript-api works (to avoid Whisper costs)
4. Limit number of videos scanned at once

## Performance Comparison

### Before (GPU-dependent)

- ✅ No API costs
- ✅ Fast processing with GPU
- ❌ Requires GPU hardware
- ❌ Slow on CPU-only servers
- ❌ No transcription for videos without subtitles

### After (OpenAI APIs)

- ✅ No GPU required
- ✅ Works on any server
- ✅ Whisper transcription for all videos
- ✅ Better embedding quality (generally)
- ✅ More powerful LLMs (GPT-4o)
- ⚠️ API costs (but predictable and scalable)
- ⚠️ Requires internet connection

## Monitoring Costs

Track your OpenAI usage at: https://platform.openai.com/usage

Set up budget alerts in your OpenAI account to avoid unexpected charges.

## Support

If you encounter issues during migration:

1. Check the logs: `logs/` directory
2. Review the Django logs for detailed error messages
3. Verify all environment variables are set correctly
4. Ensure database migration completed successfully

## Summary

This migration eliminates GPU requirements while maintaining all functionality. The main trade-off is API costs versus infrastructure costs. For most use cases, OpenAI API costs are more economical than maintaining GPU infrastructure, especially for production deployments.
