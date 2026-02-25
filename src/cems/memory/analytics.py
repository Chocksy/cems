"""Analytics operations for CEMSMemory.

NOTE: The sync analytics methods (get_stale_memories, get_hot_memories,
get_recent_memories, get_old_memories) were removed because they read from
the orphaned memory_metadata table via PostgresMetadataStore. All maintenance
jobs now use DocumentStore directly with async methods.

The AnalyticsMixin is kept as a placeholder for future async analytics methods
that will read from memory_documents via DocumentStore.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cems.memory.core import CEMSMemory

logger = logging.getLogger(__name__)


class AnalyticsMixin:
    """Mixin class providing analytics operations for CEMSMemory.

    Previously provided get_stale_memories, get_hot_memories, etc. which
    read from the orphaned memory_metadata table. Those methods were removed
    when maintenance jobs were rewritten to use DocumentStore directly.
    """

    pass
