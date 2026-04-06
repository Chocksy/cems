"""CEMS maintenance jobs for memory decay and consolidation."""

# Categories exempt from maintenance pruning/compression
PROTECTED_CATEGORIES = {"gate-rules", "guidelines", "preferences", "category-summary", "session-summary", "entity-page"}

from cems.maintenance.consolidation import ConsolidationJob
from cems.maintenance.observation_reflector import ObservationReflector
from cems.maintenance.reindex import ReindexJob
from cems.maintenance.relation_builder import RelationBuilderJob
from cems.maintenance.summarization import SummarizationJob

__all__ = ["ConsolidationJob", "SummarizationJob", "ReindexJob", "ObservationReflector", "RelationBuilderJob"]
