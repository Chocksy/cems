"""Observation reflector — consolidate overlapping observations per project.

Inspired by Mastra's Reflector Agent. Periodically reads all observations
for each project and produces a condensed, non-redundant replacement set.

Trigger: nightly (after consolidation) or on-demand via maintenance API.
"""

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from cems.llm.observation_reflection import reflect_observations

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)

# Minimum observations before triggering reflection
MIN_OBSERVATIONS_THRESHOLD = 10


class ObservationReflector:
    """Consolidate overlapping observations per project.

    Reads all observations for each project, sends to LLM for re-synthesis,
    stores consolidated set, and soft-deletes originals.
    """

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory

    async def run_async(self) -> dict:
        """Run the observation reflector across all projects.

        Returns:
            Dict with stats: projects_processed, observations_before, observations_after,
            observations_removed
        """
        doc_store = await self.memory._ensure_document_store()
        user_id = self.memory.config.user_id

        # Get all observations
        all_obs = await doc_store.get_documents_by_category(
            user_id=user_id,
            category="observation",
            limit=500,
        )

        if not all_obs:
            logger.info("No observations found to reflect")
            return {
                "projects_processed": 0,
                "observations_before": 0,
                "observations_after": 0,
                "observations_removed": 0,
            }

        # Group by source_ref (project)
        by_project: dict[str, list[dict]] = {}
        for obs in all_obs:
            key = obs.get("source_ref") or "_no_project"
            by_project.setdefault(key, []).append(obs)

        total_before = len(all_obs)
        total_after = 0
        total_removed = 0
        projects_processed = 0

        for source_ref, project_obs in by_project.items():
            if len(project_obs) < MIN_OBSERVATIONS_THRESHOLD:
                total_after += len(project_obs)
                continue

            # Sort oldest-first for the LLM (so newer info takes precedence)
            project_obs.sort(key=lambda o: o.get("created_at") or datetime.min)

            # Extract project name from source_ref for LLM context
            project_context = source_ref.replace("project:", "") if source_ref != "_no_project" else None

            logger.info(
                f"Reflecting {len(project_obs)} observations for {project_context}"
            )

            # Call LLM to consolidate (sync function, run in executor to avoid blocking)
            consolidated = await asyncio.to_thread(
                reflect_observations,
                observations=project_obs,
                project_context=project_context,
            )

            if not consolidated:
                logger.warning(f"Reflection returned empty for {project_context}, keeping originals")
                total_after += len(project_obs)
                continue

            # Sanity check: don't replace if LLM produced more than original
            if len(consolidated) >= len(project_obs):
                logger.warning(
                    f"Reflection for {project_context} produced {len(consolidated)} "
                    f"(>= {len(project_obs)} original), skipping"
                )
                total_after += len(project_obs)
                continue

            # Store consolidated observations
            actual_source_ref = source_ref if source_ref != "_no_project" else None
            stored_count = 0

            for obs in consolidated:
                try:
                    await self.memory.add_async(
                        content=obs["content"],
                        scope="personal",
                        category="observation",
                        tags=["observation", obs.get("priority", "medium"), "reflected"],
                        infer=False,
                        source_ref=actual_source_ref,
                    )
                    stored_count += 1
                except Exception as e:
                    logger.error(f"Failed to store reflected observation: {e}")

            # Guard: only delete originals if ALL consolidated were stored
            if stored_count < len(consolidated):
                logger.error(
                    f"Only stored {stored_count}/{len(consolidated)} reflected observations "
                    f"for {project_context}, keeping originals to prevent data loss"
                )
                total_after += len(project_obs)
                continue

            # Soft-delete originals (safe — all replacements stored)
            deleted_count = 0
            for original in project_obs:
                doc_id = original.get("id")
                if doc_id:
                    try:
                        await doc_store.delete_document(doc_id, hard=False)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to soft-delete observation {doc_id}: {e}")

            logger.info(
                f"Reflected {project_context}: {len(project_obs)} → {stored_count} "
                f"({deleted_count} soft-deleted)"
            )

            total_after += stored_count
            total_removed += deleted_count
            projects_processed += 1

        result = {
            "projects_processed": projects_processed,
            "observations_before": total_before,
            "observations_after": total_after,
            "observations_removed": total_removed,
        }
        logger.info(f"Observation reflection completed: {result}")
        return result

    def run(self) -> dict:
        """Synchronous wrapper for run_async."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run_async())
        finally:
            loop.close()
