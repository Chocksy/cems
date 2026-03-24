"""Database utilities for CEMS.

This package provides:
- FilterBuilder: Dynamic WHERE clause construction
- DocumentStore: Document + chunk storage
"""

from cems.db.document_store import DocumentStore, get_document_store
from cems.db.filter_builder import FilterBuilder

__all__ = [
    "FilterBuilder",
    "DocumentStore",
    "get_document_store",
]
