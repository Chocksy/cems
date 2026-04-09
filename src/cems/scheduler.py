"""APScheduler-based maintenance scheduler for CEMS."""

import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from cems.lib.async_utils import run_async_in_thread as _run_async
from cems.maintenance.compilation import CompilationJob
from cems.maintenance.consolidation import ConsolidationJob
from cems.maintenance.distillation import DistillationJob
from cems.maintenance.lint import LintJob
from cems.maintenance.observation_reflector import ObservationReflector
from cems.maintenance.orphan_assigner import OrphanAssignerJob
from cems.maintenance.reindex import ReindexJob
from cems.maintenance.relation_builder import RelationBuilderJob
from cems.maintenance.summarization import SummarizationJob

if TYPE_CHECKING:
    from cems.config import CEMSConfig
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)


class CEMSScheduler:
    """Background scheduler for CEMS maintenance jobs.

    Manages five scheduled maintenance jobs that run per-user:
    - Nightly (3 AM): Consolidation - merge duplicates
    - Nightly (3:15 AM): Distillation - condense verbose memories to ~500 char summaries
    - Nightly (3:30 AM): Reflection - consolidate overlapping observations
    - Weekly (Sunday 4 AM): Summarization - compress old memories, consolidate never-shown
    - Monthly (1st 5 AM): Re-indexing - rebuild embeddings, archive dead memories
    """

    def __init__(self, config: "CEMSConfig"):
        self.config = config
        self._scheduler = BackgroundScheduler()
        self._setup_jobs()

    def _get_user_ids(self) -> list[str]:
        """Get active user IDs for maintenance."""
        from cems.api.deps import get_active_user_ids

        try:
            return get_active_user_ids()
        except Exception as e:
            logger.error(f"Failed to get active user IDs: {e}")
            return []

    def _create_user_memory(self, user_id: str) -> "CEMSMemory":
        """Create a per-user CEMSMemory instance."""
        from cems.api.deps import create_user_memory

        return create_user_memory(user_id)

    def _setup_jobs(self) -> None:
        """Set up all scheduled jobs."""
        self._scheduler.add_job(
            self._run_consolidation,
            CronTrigger(hour=self.config.nightly_hour),
            id="nightly_consolidation",
            name="Nightly Consolidation",
            replace_existing=True,
        )

        self._scheduler.add_job(
            self._run_distillation,
            CronTrigger(hour=self.config.nightly_hour, minute=15),
            id="nightly_distillation",
            name="Nightly Distillation",
            replace_existing=True,
        )

        reflect_hour = self.config.nightly_hour
        self._scheduler.add_job(
            self._run_reflection,
            CronTrigger(hour=reflect_hour, minute=30),
            id="nightly_reflection",
            name="Nightly Observation Reflection",
            replace_existing=True,
        )

        self._scheduler.add_job(
            self._run_summarization,
            CronTrigger(
                day_of_week=self.config.weekly_day,
                hour=self.config.weekly_hour,
            ),
            id="weekly_summarization",
            name="Weekly Summarization",
            replace_existing=True,
        )

        self._scheduler.add_job(
            self._run_reindex,
            CronTrigger(
                day=self.config.monthly_day,
                hour=self.config.monthly_hour,
            ),
            id="monthly_reindex",
            name="Monthly Re-indexing",
            replace_existing=True,
        )

        # Knowledge Engine jobs
        self._scheduler.add_job(
            self._run_relations,
            CronTrigger(hour="*/6", minute=30),
            id="relation_builder",
            name="Relation Builder (6h)",
            replace_existing=True,
        )

        self._scheduler.add_job(
            self._run_compilation,
            CronTrigger(hour="*/6"),
            id="entity_compilation",
            name="Entity Compilation (6h)",
            replace_existing=True,
        )

        # Lint: daily at 4:30 AM
        self._scheduler.add_job(
            self._run_lint,
            CronTrigger(hour=self.config.nightly_hour, minute=45),
            id="daily_lint",
            name="Daily Knowledge Lint",
            replace_existing=True,
        )

        # Orphan assigner: daily at 4:55 AM (avoids */10 relation_builder/compilation)
        self._scheduler.add_job(
            self._run_orphan_assigner,
            CronTrigger(hour=self.config.nightly_hour, minute=55),
            id="daily_orphan_assigner",
            name="Daily Orphan Assigner",
            replace_existing=True,
        )

        logger.info("Maintenance jobs scheduled")

    def _run_for_all_users(self, job_type: str) -> None:
        """Run a maintenance job for all active users."""
        logger.info(f"Starting scheduled {job_type}...")
        user_ids = self._get_user_ids()
        if not user_ids:
            logger.info(f"No active users found, skipping {job_type}")
            return

        for user_id in user_ids:
            memory = None
            try:
                memory = self._create_user_memory(user_id)
                result = self._run_job_for_memory(job_type, memory)
                logger.info(f"{job_type} for user {user_id[:8]}: {result}")
            except Exception as e:
                logger.error(f"{job_type} failed for user {user_id[:8]}: {e}")
            finally:
                # Close the document store pool to prevent connection leak
                if memory and hasattr(memory, '_document_store') and memory._document_store:
                    try:
                        _run_async(memory._document_store.close())
                    except Exception:
                        pass

    def _run_consolidation(self) -> None:
        self._run_for_all_users("consolidation")

    def _run_distillation(self) -> None:
        self._run_for_all_users("distillation")

    def _run_reflection(self) -> None:
        self._run_for_all_users("reflect")

    def _run_summarization(self) -> None:
        self._run_for_all_users("summarization")

    def _run_reindex(self) -> None:
        self._run_for_all_users("reindex")

    def _run_relations(self) -> None:
        self._run_for_all_users("relations")

    def _run_compilation(self) -> None:
        self._run_for_all_users("compilation")

    def _run_lint(self) -> None:
        self._run_for_all_users("lint")

    def _run_orphan_assigner(self) -> None:
        self._run_for_all_users("orphan_assigner")

    def start(self) -> None:
        """Start the scheduler."""
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info("CEMS scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("CEMS scheduler stopped")

    def run_now(
        self, job_type: str, memory: "CEMSMemory | None" = None, **kwargs
    ) -> dict:
        """Run a maintenance job immediately.

        Args:
            job_type: One of "consolidation", "summarization", "reindex", "reflect"
            memory: Optional per-user memory instance. If provided, runs for
                    that user only. If None, runs for all active users.
            **kwargs: Extra params passed to the job (e.g. full_sweep, limit, offset)

        Returns:
            Job result dict (single user) or dict of user_id -> result (all users)
        """
        valid_jobs = {"consolidation", "distillation", "summarization", "reindex", "reflect", "relations", "compilation", "lint", "orphan_assigner"}
        if job_type not in valid_jobs:
            raise ValueError(f"Unknown job type: {job_type}. Use: {sorted(valid_jobs)}")

        if memory:
            return self._run_job_for_memory(job_type, memory, **kwargs)

        user_ids = self._get_user_ids()
        if not user_ids:
            return {"skipped": "no active users"}

        results = {}
        for user_id in user_ids:
            try:
                user_memory = self._create_user_memory(user_id)
                results[user_id[:8]] = self._run_job_for_memory(
                    job_type, user_memory, **kwargs
                )
            except Exception as e:
                results[user_id[:8]] = {"error": str(e)}
        return results

    def _run_job_for_memory(self, job_type: str, memory: "CEMSMemory", **kwargs) -> dict:
        """Run a specific job type with a given memory instance."""
        jobs = {
            "consolidation": lambda: ConsolidationJob(memory).run_async(**kwargs),
            "distillation": lambda: DistillationJob(memory).run_async(),
            "summarization": lambda: SummarizationJob(memory).run_async(),
            "reindex": lambda: ReindexJob(memory).run_async(),
            "reflect": lambda: ObservationReflector(memory).run_async(),
            "relations": lambda: RelationBuilderJob(memory).run_async(limit=100),
            "compilation": lambda: CompilationJob(memory).run_async(limit=20),
            "lint": lambda: LintJob(memory).run_async(detect_contradictions=False),
            "orphan_assigner": lambda: OrphanAssignerJob(memory).run_async(**kwargs),
        }
        return _run_async(jobs[job_type]())

    def get_jobs(self) -> list[dict]:
        """Get list of scheduled jobs."""
        jobs = []
        for job in self._scheduler.get_jobs():
            next_run = getattr(job, "next_run_time", None)
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": next_run.isoformat() if next_run else None,
            })
        return jobs

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._scheduler.running


def create_scheduler(config: "CEMSConfig") -> CEMSScheduler:
    """Create and return a CEMS scheduler."""
    return CEMSScheduler(config)
