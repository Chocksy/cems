"""Nightly distillation job — condense verbose memories into terse summaries.

Philosophy: every memory gets two versions:
- content (~500 chars) — condensed summary, always returned in search/agentic
- content_detailed (full original) — fetched on-demand via /recall

Uses DocumentStore (memory_documents) exclusively via async pattern.
"""

import logging
from typing import TYPE_CHECKING

from cems.llm.client import get_client

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)

# Memories with content longer than this get distilled
DISTILLATION_THRESHOLD = 500  # chars

# Cap content_detailed growth (condensed if exceeded)
DETAILED_CAP = 10_000  # chars

# Entity pages are excluded — they are compiled wiki documents that must keep
# their full content for search/agentic retrieval. Only 'pinned' tag and
# 'entity-page' category are protected; all other categories are eligible.

DISTILLATION_MODEL = "google/gemini-2.5-flash-lite"

DISTILLATION_PROMPT = """You are condensing a memory into a terse, fact-dense summary.

Rules:
- Maximum 500 characters
- Preserve exact values: names, numbers, dates, versions, handles, prices
- Use terse language — no filler words, no articles where unnecessary
- Frame state changes explicitly ("switched from X to Y", "replaced X with Y")
- Each fact should be a complete standalone sentence
- Preserve project context (e.g., "In Chocksy/cems: ...")
- When multiple facts exist, use semicolons or line breaks to separate
- Priority order: decisions > preferences > events > observations

Input memory (category: {category}):
{content}

Output: condensed summary (plain text, no JSON, no markdown headers)"""

DETAILED_CONDENSATION_PROMPT = """Condense this detailed memory content to under {cap} characters while preserving all unique facts, decisions, and specific values.

Rules:
- Keep all names, numbers, dates, versions, handles
- Remove redundancy and filler
- Merge overlapping facts
- Preserve project references

Content:
{content}

Output: condensed text (plain text, no JSON)"""


class DistillationJob:
    """Nightly job to condense verbose memories into terse summaries.

    Finds memories where content > DISTILLATION_THRESHOLD chars and distills them:
    1. Copies current content → content_detailed (preserves original)
    2. LLM-condenses content to ~500 chars (terse, observer-style)
    3. Atomically updates both columns via distill_document()
    """

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory
        self.config = memory.config

    async def run_async(self) -> dict[str, int]:
        """Run the distillation job.

        Returns:
            Dict with counts: distilled, skipped, errors, candidates
        """
        doc_store = await self.memory._ensure_document_store()
        user_id = self.config.user_id

        # Oldest first so we distill the oldest content first
        all_docs = await doc_store.get_all_documents(user_id, limit=2000, order="asc")

        # Find candidates: content > threshold, not pinned, not entity pages
        candidates = [
            d for d in all_docs
            if len(d.get("content", "")) > DISTILLATION_THRESHOLD
            and "pinned" not in (d.get("tags") or [])
            and d.get("category") != "entity-page"
        ]

        logger.info(
            f"Distillation: {len(candidates)} candidates out of {len(all_docs)} total"
        )

        distilled = 0
        skipped = 0
        errors = 0

        for doc in candidates:
            doc_id = doc["id"]
            content = doc["content"]
            existing_detailed = doc.get("content_detailed")

            # Determine what to store in content_detailed
            # If already has content_detailed, keep it (don't overwrite with already-condensed content)
            # If no content_detailed, the current content IS the original — preserve it
            detailed = existing_detailed or content

            # Cap content_detailed growth
            if len(detailed) > DETAILED_CAP:
                try:
                    detailed = await self._condense_detailed(detailed)
                except Exception as e:
                    logger.error(f"Failed to condense content_detailed for {doc_id[:8]}: {e}")

            # LLM distillation of content
            try:
                summary = await self._distill(content, doc.get("category", "general"))
            except Exception as e:
                logger.error(f"Distillation LLM call failed for {doc_id[:8]}: {e}")
                errors += 1
                continue

            if not summary or len(summary) >= len(content):
                # LLM didn't actually condense — skip
                skipped += 1
                continue

            # Atomic update: content (condensed) + content_detailed (original)
            try:
                success = await doc_store.distill_document(
                    document_id=doc_id,
                    user_id=user_id,
                    content=summary,
                    content_detailed=detailed,
                )
                if success:
                    # Re-chunk and re-embed the condensed content so search
                    # embeddings match the new content (prevents stale chunks)
                    from cems.chunking import chunk_document
                    new_chunks = chunk_document(summary)
                    chunk_texts = [c.content for c in new_chunks]
                    await self.memory._ensure_initialized_async()
                    new_embeddings = await self.memory._async_embedder.embed_batch(chunk_texts)
                    await doc_store.refresh_chunks(doc_id, new_chunks, new_embeddings)

                    distilled += 1
                    logger.debug(
                        f"Distilled {doc_id[:8]}: {len(content)} → {len(summary)} chars "
                        f"({len(new_chunks)} chunks re-embedded)"
                    )
                else:
                    errors += 1
            except Exception as e:
                logger.error(f"Failed to store distilled doc {doc_id[:8]}: {e}")
                errors += 1

        result = {
            "distilled": distilled,
            "skipped": skipped,
            "errors": errors,
            "candidates": len(candidates),
            "total_checked": len(all_docs),
        }
        logger.info(f"Distillation completed: {result}")
        return result

    async def _distill(self, content: str, category: str) -> str | None:
        """Distill content into a terse ~500 char summary via LLM."""
        client = get_client()

        prompt = DISTILLATION_PROMPT.format(
            content=content[:3000],  # Cap input to avoid huge prompts
            category=category,
        )

        try:
            response = client.complete(
                prompt=prompt,
                system="You are a memory condensation agent. Output only the condensed text.",
                model=DISTILLATION_MODEL,
                temperature=0.1,
                max_tokens=1000,
                fast_route=False,
            )
            return response.strip() if response else None
        except Exception as e:
            logger.error(f"Distillation LLM error: {e}")
            raise

    async def _condense_detailed(self, content: str) -> str:
        """Condense content_detailed when it exceeds DETAILED_CAP."""
        client = get_client()
        target = DETAILED_CAP // 2  # Condense to half the cap

        prompt = DETAILED_CONDENSATION_PROMPT.format(
            content=content[:DETAILED_CAP * 2],  # Cap LLM input
            cap=target,
        )

        try:
            response = client.complete(
                prompt=prompt,
                system="You are a memory condensation agent. Output only the condensed text.",
                model=DISTILLATION_MODEL,
                temperature=0.1,
                max_tokens=4000,
                fast_route=False,
            )
            return response.strip() if response else content[:DETAILED_CAP]
        except Exception:
            # Fallback: truncate at line boundary
            truncated = content[:DETAILED_CAP]
            last_newline = truncated.rfind("\n", DETAILED_CAP // 2)
            if last_newline > 0:
                return truncated[:last_newline]
            return truncated
