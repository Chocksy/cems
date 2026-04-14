#!/usr/bin/env python3
"""One-time cleanup of duplicate entity pages.

Fetches all entity pages via the CEMS API, groups them by title similarity
(word overlap > 60%), and merges duplicates by keeping the page with the
highest shown_count and soft-deleting the rest.

Usage:
    python scripts/merge_duplicate_entities.py                          # Dry run on cems.chocksy.com
    python scripts/merge_duplicate_entities.py --execute                # Apply merges
    python scripts/merge_duplicate_entities.py --api-url http://localhost:8765  # Different instance
"""

import argparse
import os
import re
import sys
from pathlib import Path

import requests


def get_default_credentials():
    """Read API URL and key from ~/.cems/credentials."""
    creds_file = Path.home() / ".cems" / "credentials"
    url = "https://cems.chocksy.com"
    key = None
    if creds_file.exists():
        for line in creds_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("CEMS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
            if line.startswith("CEMS_API_URL="):
                url = line.split("=", 1)[1].strip().strip('"')
    return url, key


def api_get(base_url, path, api_key, params=None):
    """GET request to CEMS API."""
    resp = requests.get(
        f"{base_url}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        params=params or {},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def api_delete(base_url, path, api_key):
    """DELETE request to CEMS API."""
    resp = requests.delete(
        f"{base_url}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def normalize_title(title):
    """Normalize title to a set of lowercase words for comparison."""
    return set(re.sub(r"[^a-z0-9 ]", "", title.lower().strip()).split())


def title_similarity(title_a, title_b):
    """Word overlap similarity between two titles.

    Returns overlap / min(len_a, len_b) — high value means the shorter
    title's words are mostly contained in the longer one.
    """
    words_a = normalize_title(title_a)
    words_b = normalize_title(title_b)
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a & words_b)
    return overlap / min(len(words_a), len(words_b))


def main():
    parser = argparse.ArgumentParser(description="Merge duplicate entity pages")
    parser.add_argument("--execute", action="store_true", help="Apply merges (default: dry run)")
    parser.add_argument("--api-url", help="CEMS API URL (default: from credentials)")
    parser.add_argument("--api-key", help="CEMS API key (default: from credentials)")
    parser.add_argument(
        "--threshold", type=float, default=0.60,
        help="Word overlap threshold (default: 0.60 = 60%%)",
    )
    parser.add_argument(
        "--min-overlap", type=int, default=3,
        help="Minimum shared words required (default: 3)",
    )
    args = parser.parse_args()

    default_url, default_key = get_default_credentials()
    base_url = args.api_url or os.environ.get("CEMS_API_URL") or default_url
    api_key = args.api_key or os.environ.get("CEMS_API_KEY") or default_key

    if not api_key:
        print("No API key found. Set CEMS_API_KEY or use --api-key.")
        sys.exit(1)

    print(f"Target: {base_url}")
    print(f"Threshold: {args.threshold} ({args.threshold*100:.0f}% word overlap)")
    print(f"Min shared words: {args.min_overlap}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print()

    # Fetch all entity pages
    data = api_get(base_url, "/api/wiki/entities", api_key, {"limit": 200})
    entities = data.get("entities", [])
    print(f"Found {len(entities)} entity pages")

    if len(entities) < 2:
        print("Not enough entity pages to check for duplicates.")
        return

    # Find duplicate groups by title similarity
    merged_ids = set()
    merge_groups = []

    for i in range(len(entities)):
        eid_i = entities[i]["id"]
        if eid_i in merged_ids:
            continue
        title_i = entities[i].get("title", "")
        group = [entities[i]]
        for j in range(i + 1, len(entities)):
            eid_j = entities[j]["id"]
            if eid_j in merged_ids:
                continue
            title_j = entities[j].get("title", "")
            words_overlap = len(normalize_title(title_i) & normalize_title(title_j))
            sim = title_similarity(title_i, title_j)
            if sim >= args.threshold and words_overlap >= args.min_overlap:
                group.append(entities[j])
                merged_ids.add(eid_j)
        if len(group) > 1:
            merge_groups.append(group)

    total_to_merge = sum(len(g) - 1 for g in merge_groups)
    print(f"\nFound {len(merge_groups)} duplicate groups ({total_to_merge} pages to soft-delete):")
    print()

    for i, group in enumerate(merge_groups):
        # Sort by shown_count desc — keep the one with highest
        group.sort(key=lambda e: e.get("shown_count", 0) or 0, reverse=True)
        keep = group[0]
        discard = group[1:]

        print(f"Group {i+1} ({len(group)} pages):")
        print(f"  KEEP:   \"{keep['title']}\" (id={keep['id'][:12]}, shown={keep.get('shown_count', 0)}, created={keep.get('created_at', '')[:10]})")
        for d in discard:
            print(f"  DELETE: \"{d['title']}\" (id={d['id'][:12]}, shown={d.get('shown_count', 0)}, created={d.get('created_at', '')[:10]})")

        if args.execute:
            for d in discard:
                try:
                    api_delete(base_url, f"/api/memory/{d['id']}", api_key)
                    print(f"    -> Soft-deleted {d['id'][:12]}")
                except Exception as e:
                    print(f"    -> Failed to delete {d['id'][:12]}: {e}")
        print()

    if not args.execute:
        print(f"Dry run complete. Pass --execute to apply {total_to_merge} merges.")
    else:
        print(f"Done. Soft-deleted {total_to_merge} duplicate entity pages.")


if __name__ == "__main__":
    main()
