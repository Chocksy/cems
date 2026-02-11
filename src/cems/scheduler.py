"""APScheduler-based maintenance scheduler for CEMS."""

import logging
from typing import TYPE_CHECKING, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from cems.maintenance.consolidation import ConsolidationJob
from cems.maintenance.observation_reflector import ObservationReflector
from cems.maintenance.reindex import ReindexJob
from cems.maintenance.summarization import SummarizationJob

if TYPE_CHECKING:
    from cems.config import CEMSConfig
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)


class CEMSScheduler:
    """Background scheduler for CEMS maintenance jobs.

    Manages three scheduled maintenance jobs:
    - Nightly (3 AM): Consolidation - merge duplicates, promote hot memories
    - Weekly (Sunday 4 AM): Summarization - compress old memories, prune stale
    - Monthly (1st 5 AM): Re-indexing - rebuild embeddings, archive dead memories
    """

    def __init__(self, memory: "CEMSMemory"):
        """Initialize the scheduler.

        Args:
            memory: CEMSMemory instance to run maintenance on
        """
        self.memory = memory
        self.config = memory.config
        self._scheduler = BackgroundScheduler()
        self._setup_jobs()

    def _setup_jobs(self) -> None:
        """Set up all scheduled jobs."""
        # Nightly consolidation
        self._scheduler.add_job(
            self._run_consolidation,
            CronTrigger(hour=self.config.nightly_hour),
            id="nightly_consolidation",
            name="Nightly Consolidation",
            replace_existing=True,
        )

        # Nightly observation reflection (runs 30 min after consolidation)
        reflect_hour = self.config.nightly_hour
        self._scheduler.add_job(
            self._run_reflection,
            CronTrigger(hour=reflect_hour, minute=30),
            id="nightly_reflection",
            name="Nightly Observation Reflection",
            replace_existing=True,
        )

        # Weekly summarization
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

        # Monthly re-indexing
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

        logger.info("Maintenance jobs scheduled")

    def _run_consolidation(self) -> None:
        """Run the nightly consolidation job."""
        logger.info("Starting nightly consolidation...")
        try:
            job = ConsolidationJob(self.memory)
            result = job.run()
            logger.info(f"Nightly consolidation completed: {result}")
        except Exception as e:
            logger.error(f"Nightly consolidation failed: {e}")

    def _run_reflection(self) -> None:
        """Run the nightly observation reflection job."""
        import asyncio

        logger.info("Starting nightly observation reflection...")
        try:
            reflector = ObservationReflector(self.memory)
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(reflector.run_async())
            finally:
                loop.close()
            logger.info(f"Nightly observation reflection completed: {result}")
        except Exception as e:
            logger.error(f"Nightly observation reflection failed: {e}")

    def _run_summarization(self) -> None:
        """Run the weekly summarization job."""
        logger.info("Starting weekly summarization...")
        try:
            job = SummarizationJob(self.memory)
            result = job.run()
            logger.info(f"Weekly summarization completed: {result}")
        except Exception as e:
            logger.error(f"Weekly summarization failed: {e}")

    def _run_reindex(self) -> None:
        """Run the monthly re-indexing job."""
        logger.info("Starting monthly re-indexing...")
        try:
            job = ReindexJob(self.memory)
            result = job.run()
            logger.info(f"Monthly re-indexing completed: {result}")
        except Exception as e:
            logger.error(f"Monthly re-indexing failed: {e}")

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

    def run_now(self, job_type: str) -> dict:
        """Run a maintenance job immediately.

        Args:
            job_type: One of "consolidation", "summarization", "reindex"

        Returns:
            Job result dict
        """
        jobs = {
            "consolidation": self._run_consolidation_sync,
            "summarization": self._run_summarization_sync,
            "reindex": self._run_reindex_sync,
        }

        if job_type not in jobs:
            raise ValueError(f"Unknown job type: {job_type}. Use: {list(jobs.keys())}")

        return jobs[job_type]()

    def _run_consolidation_sync(self) -> dict:
        """Run consolidation synchronously and return result."""
        job = ConsolidationJob(self.memory)
        return job.run()

    def _run_summarization_sync(self) -> dict:
        """Run summarization synchronously and return result."""
        job = SummarizationJob(self.memory)
        return job.run()

    def _run_reindex_sync(self) -> dict:
        """Run re-indexing synchronously and return result."""
        job = ReindexJob(self.memory)
        return job.run()

    def get_jobs(self) -> list[dict]:
        """Get list of scheduled jobs.

        Returns:
            List of job info dicts
        """
        jobs = []
        for job in self._scheduler.get_jobs():
            next_run = job.next_run_time
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


def create_scheduler(memory: "CEMSMemory") -> CEMSScheduler:
    """Create and return a CEMS scheduler.

    Args:
        memory: CEMSMemory instance

    Returns:
        CEMSScheduler instance (not started)
    """
    return CEMSScheduler(memory)
