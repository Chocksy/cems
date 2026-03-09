-- Migration: Add relevance feedback counters to memory_documents
-- Run after: migrate_soft_delete_feedback.sql
--
-- These counters track how often Claude reports a memory as relevant vs noise,
-- closing the feedback loop from the memory-relevance.md rule.

ALTER TABLE memory_documents ADD COLUMN IF NOT EXISTS relevant_count INT NOT NULL DEFAULT 0;
ALTER TABLE memory_documents ADD COLUMN IF NOT EXISTS noise_count INT NOT NULL DEFAULT 0;
