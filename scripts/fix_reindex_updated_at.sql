-- Fix updated_at timestamps that were mass-reset by the reindex job.
-- The reindex job called update_document() which bumps updated_at = NOW(),
-- defeating all age-based pruning (stale_days, archive_days).
-- This restores updated_at to created_at for docs that were only reindexed,
-- not genuinely updated.
--
-- Run: psql -U cems -d cems -f fix_reindex_updated_at.sql

-- Fix March 3 mass reindex (3306 docs affected)
UPDATE memory_documents
SET updated_at = created_at
WHERE deleted_at IS NULL
  AND DATE(updated_at) = '2026-03-03'
  AND created_at < updated_at - INTERVAL '2 days';

-- Fix March 5 mass reindex (1066 docs affected)
UPDATE memory_documents
SET updated_at = created_at
WHERE deleted_at IS NULL
  AND DATE(updated_at) = '2026-03-05'
  AND created_at < updated_at - INTERVAL '2 days';

-- Fix March 6 mass reindex (698 docs affected)
UPDATE memory_documents
SET updated_at = created_at
WHERE deleted_at IS NULL
  AND DATE(updated_at) = '2026-03-06'
  AND created_at < updated_at - INTERVAL '2 days';

-- Verify: should show a spread of updated_at dates now, not just March 3+
SELECT
  CASE WHEN updated_at > NOW() - INTERVAL '7 days' THEN '0-7d'
       WHEN updated_at > NOW() - INTERVAL '30 days' THEN '7-30d'
       WHEN updated_at > NOW() - INTERVAL '60 days' THEN '30-60d'
       ELSE '60d+' END as age_bucket,
  COUNT(*)
FROM memory_documents WHERE deleted_at IS NULL
GROUP BY age_bucket ORDER BY age_bucket;
