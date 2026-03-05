#!/usr/bin/env python3
"""Analyze memory health, distribution, and quality.

Standalone read-only script — uses asyncpg directly, no CEMS imports needed.

Usage:
    CEMS_DATABASE_URL="postgresql://..." python scripts/analyze_memories.py
    CEMS_DATABASE_URL="postgresql://..." python scripts/analyze_memories.py --user-id UUID
"""

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime

import asyncpg


def _fmt_bytes(n: int | None) -> str:
    if not n:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


async def overview(conn, user_filter: str, params: list):
    _section("Overview")
    row = await conn.fetchrow(f"""
        SELECT
            COUNT(*) FILTER (WHERE deleted_at IS NULL) AS active,
            COUNT(*) FILTER (WHERE deleted_at IS NOT NULL) AS deleted,
            MIN(created_at) AS earliest,
            MAX(created_at) AS latest,
            SUM(content_bytes) FILTER (WHERE deleted_at IS NULL) AS total_bytes
        FROM memory_documents
        {user_filter}
    """, *params)
    chunks = await conn.fetchval(f"""
        SELECT COUNT(*) FROM memory_chunks c
        JOIN memory_documents d ON c.document_id = d.id
        {user_filter} AND d.deleted_at IS NULL
    """.replace("WHERE", "WHERE", 1), *params)
    print(f"  Active documents:  {row['active']}")
    print(f"  Deleted documents: {row['deleted']}")
    print(f"  Active chunks:     {chunks}")
    print(f"  Date range:        {row['earliest']} → {row['latest']}")
    print(f"  Total storage:     {_fmt_bytes(row['total_bytes'])}")


async def category_distribution(conn, user_filter: str, params: list):
    _section("Category Distribution")
    # Canonical categories from learning_extraction.py
    canonical = {
        "api", "architecture", "auth", "config",
        "database", "debugging", "deployment", "documentation",
        "frontend", "general", "infrastructure",
        "monitoring", "networking", "performance",
        "refactoring", "security", "session-summary",
        "testing", "workflow",
    }
    rows = await conn.fetch(f"""
        SELECT category, COUNT(*) AS cnt,
               COALESCE(SUM(content_bytes), 0) AS bytes
        FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NULL
        GROUP BY category ORDER BY cnt DESC
    """, *params)
    non_canonical = []
    for r in rows:
        marker = ""
        cat = r["category"] or "(null)"
        if cat not in canonical and cat not in (
            "gate-rules", "guidelines", "preferences", "category-summary",
        ):
            marker = " [NON-CANONICAL]"
            non_canonical.append(cat)
        print(f"  {cat:30s}  {r['cnt']:5d}  {_fmt_bytes(r['bytes']):>10s}{marker}")
    if non_canonical:
        print(f"\n  WARNING: {len(non_canonical)} non-canonical categories found")


async def project_distribution(conn, user_filter: str, params: list):
    _section("Project Distribution")
    rows = await conn.fetch(f"""
        SELECT COALESCE(source_ref, '(none)') AS ref, COUNT(*) AS cnt,
               COALESCE(SUM(content_bytes), 0) AS bytes
        FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NULL
        GROUP BY source_ref ORDER BY cnt DESC
        LIMIT 20
    """, *params)
    for r in rows:
        print(f"  {r['ref']:50s}  {r['cnt']:5d}  {_fmt_bytes(r['bytes']):>10s}")


async def shown_distribution(conn, user_filter: str, params: list):
    _section("Shown Distribution")
    row = await conn.fetchrow(f"""
        SELECT
            COUNT(*) FILTER (WHERE shown_count = 0) AS never_shown,
            COUNT(*) FILTER (WHERE shown_count BETWEEN 1 AND 5) AS shown_1_5,
            COUNT(*) FILTER (WHERE shown_count BETWEEN 6 AND 20) AS shown_6_20,
            COUNT(*) FILTER (WHERE shown_count > 20) AS shown_20_plus,
            AVG(shown_count)::numeric(10,1) AS avg_shown,
            MAX(shown_count) AS max_shown
        FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NULL
    """, *params)
    print(f"  Never shown:   {row['never_shown']}")
    print(f"  Shown 1-5x:    {row['shown_1_5']}")
    print(f"  Shown 6-20x:   {row['shown_6_20']}")
    print(f"  Shown 20+x:    {row['shown_20_plus']}")
    print(f"  Average:       {row['avg_shown']}")
    print(f"  Maximum:       {row['max_shown']}")

    # Never-shown by age bucket
    age_rows = await conn.fetch(f"""
        SELECT
            CASE
                WHEN created_at > NOW() - INTERVAL '7 days' THEN '0-7d'
                WHEN created_at > NOW() - INTERVAL '30 days' THEN '7-30d'
                WHEN created_at > NOW() - INTERVAL '90 days' THEN '30-90d'
                ELSE '90d+'
            END AS age_bucket,
            COUNT(*) AS cnt
        FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NULL
            AND shown_count = 0
        GROUP BY age_bucket ORDER BY MIN(created_at) DESC
    """, *params)
    if age_rows:
        print("\n  Never-shown by age:")
        for r in age_rows:
            print(f"    {r['age_bucket']:8s}  {r['cnt']:5d}")


async def age_distribution(conn, user_filter: str, params: list):
    _section("Age Distribution")
    rows = await conn.fetch(f"""
        SELECT
            CASE
                WHEN created_at > NOW() - INTERVAL '7 days' THEN '0-7d'
                WHEN created_at > NOW() - INTERVAL '30 days' THEN '7-30d'
                WHEN created_at > NOW() - INTERVAL '90 days' THEN '30-90d'
                ELSE '90d+'
            END AS age_bucket,
            COUNT(*) AS cnt,
            COALESCE(SUM(content_bytes), 0) AS bytes
        FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NULL
        GROUP BY age_bucket ORDER BY MIN(created_at) DESC
    """, *params)
    for r in rows:
        print(f"  {r['age_bucket']:8s}  {r['cnt']:5d}  {_fmt_bytes(r['bytes']):>10s}")


async def storage_analysis(conn, user_filter: str, params: list):
    _section("Storage Analysis (Top categories by bytes)")
    rows = await conn.fetch(f"""
        SELECT category, COALESCE(SUM(content_bytes), 0) AS bytes,
               COUNT(*) AS cnt
        FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NULL
        GROUP BY category ORDER BY bytes DESC
        LIMIT 10
    """, *params)
    for r in rows:
        cat = r["category"] or "(null)"
        print(f"  {cat:30s}  {_fmt_bytes(r['bytes']):>10s}  ({r['cnt']} docs)")


async def content_quality(conn, user_filter: str, params: list):
    _section("Content Quality")
    row = await conn.fetchrow(f"""
        SELECT
            COUNT(*) FILTER (WHERE LENGTH(content) < 50) AS tiny,
            COUNT(*) FILTER (WHERE LENGTH(content) BETWEEN 50 AND 500) AS small,
            COUNT(*) FILTER (WHERE LENGTH(content) BETWEEN 501 AND 2000) AS medium,
            COUNT(*) FILTER (WHERE LENGTH(content) BETWEEN 2001 AND 5000) AS large,
            COUNT(*) FILTER (WHERE LENGTH(content) > 5000) AS huge
        FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NULL
    """, *params)
    print(f"  <50 chars:     {row['tiny']}")
    print(f"  50-500:        {row['small']}")
    print(f"  500-2000:      {row['medium']}")
    print(f"  2000-5000:     {row['large']}")
    print(f"  5000+:         {row['huge']}")

    # Top 5 longest docs
    top = await conn.fetch(f"""
        SELECT id, category, LENGTH(content) AS len, LEFT(title, 50) AS title_short
        FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NULL
        ORDER BY LENGTH(content) DESC LIMIT 5
    """, *params)
    print("\n  Top 5 largest documents:")
    for r in top:
        title = r["title_short"] or "(no title)"
        print(f"    {r['id']}  {r['category']:20s}  {r['len']:6d} chars  {title}")


async def tag_analysis(conn, user_filter: str, params: list):
    _section("Tag Analysis")
    rows = await conn.fetch(f"""
        SELECT tag, COUNT(*) AS cnt
        FROM memory_documents, LATERAL unnest(tags) AS tag
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NULL
        GROUP BY tag ORDER BY cnt DESC
        LIMIT 20
    """, *params)
    for r in rows:
        print(f"  {r['tag']:40s}  {r['cnt']:5d}")

    # Special tag counts
    special = await conn.fetchrow(f"""
        SELECT
            COUNT(*) FILTER (WHERE tags && ARRAY['session-summary']) AS session_tags,
            COUNT(*) FILTER (WHERE EXISTS (
                SELECT 1 FROM unnest(tags) t WHERE t LIKE 'session:%%'
            )) AS session_prefix_tags,
            COUNT(*) FILTER (WHERE tags && ARRAY['tool-learning']) AS tool_learning_tags,
            COUNT(*) FILTER (WHERE tags && ARRAY['category-summary']) AS cat_summary_tags
        FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NULL
    """, *params)
    print(f"\n  Session tags:          {special['session_tags']}")
    print(f"  Session prefix (session:*): {special['session_prefix_tags']}")
    print(f"  Tool learning tags:    {special['tool_learning_tags']}")
    print(f"  Category summary tags: {special['cat_summary_tags']}")


async def soft_delete_stats(conn, user_filter: str, params: list):
    _section("Soft-Delete Stats")
    row = await conn.fetchrow(f"""
        SELECT
            COUNT(*) AS cnt,
            COALESCE(SUM(content_bytes), 0) AS bytes,
            MIN(deleted_at) AS earliest_delete,
            MAX(deleted_at) AS latest_delete
        FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NOT NULL
    """, *params)
    print(f"  Deleted count:     {row['cnt']}")
    print(f"  Reclaimable:       {_fmt_bytes(row['bytes'])}")
    if row["earliest_delete"]:
        print(f"  Delete range:      {row['earliest_delete']} → {row['latest_delete']}")


async def recommendations(conn, user_filter: str, params: list):
    _section("Recommendations")
    recs = []

    # Check for never-shown old memories
    old_never = await conn.fetchval(f"""
        SELECT COUNT(*) FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NULL
            AND shown_count = 0
            AND created_at < NOW() - INTERVAL '30 days'
    """, *params)
    if old_never and old_never > 50:
        recs.append(
            f"  - {old_never} memories are 30+ days old and never shown. "
            "Consider reviewing/pruning these."
        )

    # Check for reclaimable deleted docs
    deleted_bytes = await conn.fetchval(f"""
        SELECT COALESCE(SUM(content_bytes), 0) FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NOT NULL
    """, *params)
    if deleted_bytes and deleted_bytes > 100_000:
        recs.append(
            f"  - {_fmt_bytes(deleted_bytes)} reclaimable from soft-deleted docs. "
            "Consider hard-deleting old deletions."
        )

    # Check for very large docs
    huge = await conn.fetchval(f"""
        SELECT COUNT(*) FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NULL
            AND LENGTH(content) > 5000
    """, *params)
    if huge and huge > 10:
        recs.append(
            f"  - {huge} documents exceed 5000 chars. "
            "Large docs may benefit from splitting or summarization."
        )

    # Check non-canonical categories
    non_canonical_count = await conn.fetchval(f"""
        SELECT COUNT(DISTINCT category) FROM memory_documents
        {user_filter} {"AND" if user_filter else "WHERE"} deleted_at IS NULL
            AND category NOT IN (
                'api', 'architecture', 'auth', 'config',
                'database', 'debugging', 'deployment', 'documentation',
                'frontend', 'general', 'infrastructure',
                'monitoring', 'networking', 'performance',
                'refactoring', 'security', 'session-summary',
                'testing', 'workflow',
                'gate-rules', 'guidelines', 'preferences', 'category-summary',
                'observation'
            )
    """, *params)
    if non_canonical_count and non_canonical_count > 0:
        recs.append(
            f"  - {non_canonical_count} non-canonical categories found. "
            "Run category normalization to clean up."
        )

    if recs:
        for r in recs:
            print(r)
    else:
        print("  No issues found. Memory health looks good!")


async def main():
    parser = argparse.ArgumentParser(description="Analyze CEMS memory health")
    parser.add_argument("--user-id", help="Filter to a specific user ID")
    args = parser.parse_args()

    db_url = os.environ.get("CEMS_DATABASE_URL")
    if not db_url:
        print("Error: CEMS_DATABASE_URL required", file=sys.stderr)
        sys.exit(1)

    conn = await asyncpg.connect(db_url)

    # Build user filter
    if args.user_id:
        user_filter = "WHERE d.user_id = $1" if False else "WHERE user_id = $1"
        params = [args.user_id]
    else:
        user_filter = ""
        params = []

    print(f"CEMS Memory Analysis — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")

    try:
        await overview(conn, user_filter, params)
        await category_distribution(conn, user_filter, params)
        await project_distribution(conn, user_filter, params)
        await shown_distribution(conn, user_filter, params)
        await age_distribution(conn, user_filter, params)
        await storage_analysis(conn, user_filter, params)
        await content_quality(conn, user_filter, params)
        await tag_analysis(conn, user_filter, params)
        await soft_delete_stats(conn, user_filter, params)
        await recommendations(conn, user_filter, params)
    finally:
        await conn.close()

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
