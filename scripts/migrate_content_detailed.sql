-- Migration: Add content_detailed column for memory distillation
-- Run after: migrate_relevance_feedback.sql
--
-- Stores the full original content when a memory is distilled (condensed).
-- The main `content` column holds the condensed summary (~500 chars).
-- `content_detailed` holds the original full text, fetchable on-demand via /recall.
-- NULL means the memory has not been distilled or is already short enough.

ALTER TABLE memory_documents ADD COLUMN IF NOT EXISTS content_detailed TEXT;
