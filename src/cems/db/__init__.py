"""Database utilities for CEMS.

This package provides:
- FilterBuilder: Dynamic WHERE clause construction
- MEMORY_COLUMNS: Standard SELECT columns for memories
- row_to_dict: Convert database rows to dictionaries
- DocumentStore: Document + chunk storage
"""

from cems.db.constants import MEMORY_COLUMNS, MEMORY_COLUMNS_PREFIXED, MEMORY_COLUMNS_WITH_SCORE
from cems.db.document_store import DocumentStore, get_document_store
from cems.db.filter_builder import FilterBuilder
from cems.db.row_mapper import row_to_dict

__all__ = [
    "MEMORY_COLUMNS",
    "MEMORY_COLUMNS_PREFIXED",
    "MEMORY_COLUMNS_WITH_SCORE",
    "FilterBuilder",
    "row_to_dict",
    "DocumentStore",
    "get_document_store",
]
