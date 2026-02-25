"""Tests for CEMS maintenance jobs.

All maintenance jobs use async + DocumentStore pattern.
Tests use AsyncMock to mock DocumentStore and memory methods.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cems.config import CEMSConfig

# Valid UUID for testing
TEST_UUID = "a6e153f9-41c5-4cbc-9a50-74160af381dd"


def _run(coro):
    """Helper to run async tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_doc(
    doc_id: str,
    content: str,
    category: str = "general",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> dict:
    """Create a mock document dict matching DocumentStore format."""
    now = datetime.now(UTC)
    return {
        "id": doc_id,
        "content": content,
        "category": category,
        "scope": "personal",
        "tags": [],
        "source": None,
        "source_ref": None,
        "created_at": created_at or now,
        "updated_at": updated_at or now,
    }


@pytest.fixture
def mock_memory():
    """Create a mock CEMSMemory with async DocumentStore support."""
    mock = MagicMock()
    mock.config = CEMSConfig(
        user_id=TEST_UUID,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )

    # Mock doc_store returned by _ensure_document_store
    doc_store = AsyncMock()
    mock._ensure_document_store = AsyncMock(return_value=doc_store)
    mock._ensure_initialized_async = AsyncMock()

    # Mock embedder
    embedder = AsyncMock()
    mock._async_embedder = embedder

    # Mock async methods
    mock.update_async = AsyncMock(return_value={"success": True, "memory_id": "test"})
    mock.add_async = AsyncMock(return_value={"id": "new-doc", "success": True})

    return mock, doc_store, embedder


class TestConsolidationJob:
    """Tests for the async ConsolidationJob with three-tier dedup."""

    def test_consolidation_no_recent_docs(self, mock_memory):
        """Returns zeros when no recent documents found."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, _ = mock_memory
        doc_store.get_recent_documents = AsyncMock(return_value=[])

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        assert result["duplicates_merged"] == 0
        assert result["conflicts_found"] == 0
        assert result["memories_checked"] == 0
        doc_store.get_recent_documents.assert_awaited_once()

    @patch("cems.maintenance.consolidation.merge_memory_contents")
    def test_tier1_automerge_above_098(self, mock_merge, mock_memory):
        """Tier 1: Auto-merges near-identical docs (>= 0.98) without LLM classify."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, embedder = mock_memory

        doc1 = _make_doc("doc-1", "Python is great for backend")
        doc2 = _make_doc("doc-2", "Python is great for backend dev")
        doc3 = _make_doc("doc-3", "I like cats")

        doc_store.get_recent_documents = AsyncMock(return_value=[doc1, doc2, doc3])
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])

        doc_store.search_chunks = AsyncMock(
            side_effect=[
                [
                    {"document_id": "doc-1", "score": 1.0},
                    {"document_id": "doc-2", "score": 0.99},  # Above 0.98 = auto-merge
                ],
                [{"document_id": "doc-3", "score": 1.0}],
            ]
        )
        doc_store.get_document = AsyncMock(return_value=doc2)
        doc_store.delete_document = AsyncMock(return_value=True)
        mock_merge.return_value = "Python is great for backend development"

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        assert result["duplicates_merged"] == 1
        assert result["llm_classifications"] == 0  # No LLM classify call
        mock_merge.assert_called_once()
        doc_store.delete_document.assert_awaited_once_with("doc-2", hard=False)

    @patch("cems.maintenance.consolidation.classify_memory_pair")
    @patch("cems.maintenance.consolidation.merge_memory_contents")
    def test_tier2_llm_classifies_duplicate(self, mock_merge, mock_classify, mock_memory):
        """Tier 2: LLM classifies pair as duplicate → merge."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, embedder = mock_memory

        doc1 = _make_doc("doc-1", "User prefers Python for backend")
        doc2 = _make_doc("doc-2", "User likes Python for backend work")

        doc_store.get_recent_documents = AsyncMock(return_value=[doc1, doc2])
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        doc_store.search_chunks = AsyncMock(
            side_effect=[
                [
                    {"document_id": "doc-1", "score": 1.0},
                    {"document_id": "doc-2", "score": 0.90},  # In LLM tier (0.80-0.98)
                ],
                [{"document_id": "doc-2", "score": 1.0}],
            ]
        )
        doc_store.get_document = AsyncMock(return_value=doc2)
        doc_store.delete_document = AsyncMock(return_value=True)

        mock_classify.return_value = {
            "classification": "duplicate",
            "explanation": "Same preference",
            "confidence": 0.92,
        }
        mock_merge.return_value = "User prefers Python for backend development"

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        assert result["duplicates_merged"] == 1
        assert result["llm_classifications"] == 1
        mock_classify.assert_called_once()
        mock_merge.assert_called_once()

    @patch("cems.maintenance.consolidation.classify_memory_pair")
    @patch("cems.maintenance.consolidation.merge_memory_contents")
    def test_tier2_llm_classifies_conflicting(self, mock_merge, mock_classify, mock_memory):
        """Tier 2: LLM classifies pair as conflicting → store conflict."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, embedder = mock_memory

        doc1 = _make_doc("doc-1", "User deployed on Hetzner")
        doc2 = _make_doc("doc-2", "User deployed on Railway")

        doc_store.get_recent_documents = AsyncMock(return_value=[doc1, doc2])
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        doc_store.search_chunks = AsyncMock(
            side_effect=[
                [
                    {"document_id": "doc-1", "score": 1.0},
                    {"document_id": "doc-2", "score": 0.88},
                ],
                [{"document_id": "doc-2", "score": 1.0}],
            ]
        )
        doc_store.get_document = AsyncMock(return_value=doc2)
        doc_store.add_conflict = AsyncMock(return_value="conflict-1")

        mock_classify.return_value = {
            "classification": "conflicting",
            "explanation": "Memory A says Hetzner, Memory B says Railway",
            "confidence": 0.88,
        }

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        assert result["conflicts_found"] == 1
        assert result["duplicates_merged"] == 0
        assert result["llm_classifications"] == 1
        doc_store.add_conflict.assert_awaited_once()
        mock_merge.assert_not_called()

    @patch("cems.maintenance.consolidation.classify_memory_pair")
    @patch("cems.maintenance.consolidation.merge_memory_contents")
    def test_tier2_llm_classifies_related_skips(self, mock_merge, mock_classify, mock_memory):
        """Tier 2: LLM classifies as related → no action."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, embedder = mock_memory

        doc1 = _make_doc("doc-1", "User prefers Python")
        doc2 = _make_doc("doc-2", "User uses pytest")

        doc_store.get_recent_documents = AsyncMock(return_value=[doc1, doc2])
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        doc_store.search_chunks = AsyncMock(
            side_effect=[
                [
                    {"document_id": "doc-1", "score": 1.0},
                    {"document_id": "doc-2", "score": 0.85},
                ],
                [{"document_id": "doc-2", "score": 1.0}],
            ]
        )
        doc_store.get_document = AsyncMock(return_value=doc2)

        mock_classify.return_value = {
            "classification": "related",
            "explanation": "Same ecosystem but different facts",
            "confidence": 0.82,
        }

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        assert result["duplicates_merged"] == 0
        assert result["conflicts_found"] == 0
        assert result["llm_classifications"] == 1
        mock_merge.assert_not_called()

    @patch("cems.maintenance.consolidation.classify_memory_pair")
    @patch("cems.maintenance.consolidation.merge_memory_contents")
    def test_tier2_low_confidence_skips(self, mock_merge, mock_classify, mock_memory):
        """Tier 2: LLM says duplicate but low confidence → no action."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, embedder = mock_memory

        doc1 = _make_doc("doc-1", "Content A")
        doc2 = _make_doc("doc-2", "Content B")

        doc_store.get_recent_documents = AsyncMock(return_value=[doc1, doc2])
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        doc_store.search_chunks = AsyncMock(
            side_effect=[
                [
                    {"document_id": "doc-1", "score": 1.0},
                    {"document_id": "doc-2", "score": 0.85},
                ],
                [{"document_id": "doc-2", "score": 1.0}],
            ]
        )
        doc_store.get_document = AsyncMock(return_value=doc2)

        mock_classify.return_value = {
            "classification": "duplicate",
            "explanation": "Maybe same?",
            "confidence": 0.5,  # Below MIN_CONFIDENCE (0.7)
        }

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        assert result["duplicates_merged"] == 0
        mock_merge.assert_not_called()

    @patch("cems.maintenance.consolidation.classify_memory_pair")
    def test_metadata_guard_different_category(self, mock_classify, mock_memory):
        """Metadata guard: different categories skip LLM classification."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, embedder = mock_memory

        doc1 = _make_doc("doc-1", "Python preference", category="preferences")
        doc2 = _make_doc("doc-2", "Python guideline", category="guidelines")

        doc_store.get_recent_documents = AsyncMock(return_value=[doc1, doc2])
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        doc_store.search_chunks = AsyncMock(
            side_effect=[
                [
                    {"document_id": "doc-1", "score": 1.0},
                    {"document_id": "doc-2", "score": 0.90},  # In LLM tier
                ],
                [{"document_id": "doc-2", "score": 1.0}],
            ]
        )
        doc_store.get_document = AsyncMock(return_value=doc2)

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        assert result["llm_classifications"] == 0  # Skipped by metadata guard
        mock_classify.assert_not_called()

    @patch("cems.maintenance.consolidation.classify_memory_pair")
    def test_metadata_guard_different_source_ref(self, mock_classify, mock_memory):
        """Metadata guard: different source_ref (projects) skip LLM classification."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, embedder = mock_memory

        doc1 = _make_doc("doc-1", "Deploy with Docker")
        doc1["source_ref"] = "project:org/repo-a"
        doc2 = _make_doc("doc-2", "Deploy with Docker compose")
        doc2["source_ref"] = "project:org/repo-b"

        doc_store.get_recent_documents = AsyncMock(return_value=[doc1, doc2])
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        doc_store.search_chunks = AsyncMock(
            side_effect=[
                [
                    {"document_id": "doc-1", "score": 1.0},
                    {"document_id": "doc-2", "score": 0.90},
                ],
                [{"document_id": "doc-2", "score": 1.0}],
            ]
        )
        doc_store.get_document = AsyncMock(return_value=doc2)

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        assert result["llm_classifications"] == 0
        mock_classify.assert_not_called()

    @patch("cems.maintenance.consolidation.merge_memory_contents")
    def test_tier3_below_llm_threshold_skips(self, mock_merge, mock_memory):
        """Tier 3: Similar docs below 0.80 are completely skipped."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, embedder = mock_memory

        doc1 = _make_doc("doc-1", "Python is great")
        doc2 = _make_doc("doc-2", "JavaScript is great")

        doc_store.get_recent_documents = AsyncMock(return_value=[doc1, doc2])
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        doc_store.search_chunks = AsyncMock(
            side_effect=[
                [{"document_id": "doc-1", "score": 1.0}, {"document_id": "doc-2", "score": 0.70}],
                [{"document_id": "doc-2", "score": 1.0}, {"document_id": "doc-1", "score": 0.70}],
            ]
        )

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        assert result["duplicates_merged"] == 0
        assert result["llm_classifications"] == 0
        mock_merge.assert_not_called()

    def test_consolidation_processes_single_doc(self, mock_memory):
        """Single document cannot have duplicates."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, _ = mock_memory

        doc1 = _make_doc("doc-1", "Only doc")
        doc_store.get_recent_documents = AsyncMock(return_value=[doc1])

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        assert result["duplicates_merged"] == 0
        assert result["memories_checked"] == 1

    @patch("cems.maintenance.consolidation.merge_memory_contents")
    def test_multi_merge_chain_updates_content(self, mock_merge, mock_memory):
        """Multi-merge chain: after merging A+B, uses merged content for A+C."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, embedder = mock_memory

        doc1 = _make_doc("doc-1", "Original content A")
        doc2 = _make_doc("doc-2", "Near-identical B")
        doc3 = _make_doc("doc-3", "Near-identical C")

        doc_store.get_recent_documents = AsyncMock(return_value=[doc1, doc2, doc3])
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])

        # doc-1 finds doc-2 and doc-3 both above automerge threshold
        doc_store.search_chunks = AsyncMock(
            side_effect=[
                [
                    {"document_id": "doc-1", "score": 1.0},
                    {"document_id": "doc-2", "score": 0.99},
                    {"document_id": "doc-3", "score": 0.99},
                ],
                [{"document_id": "doc-3", "score": 1.0}],
            ]
        )
        doc_store.get_document = AsyncMock(
            side_effect=[doc2, doc3]
        )
        doc_store.delete_document = AsyncMock(return_value=True)

        # First merge returns "A+B merged", second should use "A+B merged" + C
        mock_merge.side_effect = ["A+B merged", "A+B+C merged"]

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        assert result["duplicates_merged"] == 2
        # Second merge call should use "A+B merged" (not "Original content A")
        second_call_memories = mock_merge.call_args_list[1][1]["memories"]
        assert second_call_memories[0]["memory"] == "A+B merged"

    @patch("cems.maintenance.consolidation.classify_memory_pair")
    @patch("cems.maintenance.consolidation.merge_memory_contents")
    def test_tier2_empty_merge_skips_delete(self, mock_merge, mock_classify, mock_memory):
        """When LLM merge returns empty string, don't delete the duplicate."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, embedder = mock_memory

        doc1 = _make_doc("doc-1", "Content A")
        doc2 = _make_doc("doc-2", "Content B")

        doc_store.get_recent_documents = AsyncMock(return_value=[doc1, doc2])
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        doc_store.search_chunks = AsyncMock(
            side_effect=[
                [
                    {"document_id": "doc-1", "score": 1.0},
                    {"document_id": "doc-2", "score": 0.90},
                ],
                [{"document_id": "doc-2", "score": 1.0}],
            ]
        )
        doc_store.get_document = AsyncMock(return_value=doc2)
        doc_store.delete_document = AsyncMock(return_value=True)

        mock_classify.return_value = {
            "classification": "duplicate",
            "explanation": "Same content",
            "confidence": 0.92,
        }
        mock_merge.return_value = ""  # Empty merge result

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        assert result["duplicates_merged"] == 0  # No merge counted
        doc_store.delete_document.assert_not_awaited()  # No deletion

    @patch("cems.maintenance.consolidation.classify_memory_pair")
    def test_add_conflict_duplicate_not_counted(self, mock_classify, mock_memory):
        """Duplicate conflict (ON CONFLICT DO NOTHING) not counted."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, embedder = mock_memory

        doc1 = _make_doc("doc-1", "Content A")
        doc2 = _make_doc("doc-2", "Content B")

        doc_store.get_recent_documents = AsyncMock(return_value=[doc1, doc2])
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        doc_store.search_chunks = AsyncMock(
            side_effect=[
                [
                    {"document_id": "doc-1", "score": 1.0},
                    {"document_id": "doc-2", "score": 0.88},
                ],
                [{"document_id": "doc-2", "score": 1.0}],
            ]
        )
        doc_store.get_document = AsyncMock(return_value=doc2)
        doc_store.add_conflict = AsyncMock(return_value=None)  # Already exists

        mock_classify.return_value = {
            "classification": "conflicting",
            "explanation": "Contradictory info",
            "confidence": 0.88,
        }

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        assert result["conflicts_found"] == 0  # Not counted (already existed)
        doc_store.add_conflict.assert_awaited_once()


class TestSummarizationJob:
    """Tests for the async SummarizationJob."""

    def test_summarization_no_old_docs(self, mock_memory):
        """Returns zeros when no old documents found."""
        from cems.maintenance.summarization import SummarizationJob

        memory, doc_store, _ = mock_memory
        doc_store.get_all_documents = AsyncMock(return_value=[])

        job = SummarizationJob(memory)
        result = _run(job.run_async())

        assert result["categories_updated"] == 0
        assert result["memories_pruned"] == 0
        assert result["old_memories_checked"] == 0

    @patch("cems.llm.summarize_memories")
    def test_summarization_compresses_category(self, mock_summarize, mock_memory):
        """Compresses categories with 3+ old docs into summaries."""
        from cems.maintenance.summarization import SummarizationJob

        memory, doc_store, _ = mock_memory
        mock_summarize.return_value = "Summary of debugging tips"

        old_time = datetime.now(UTC) - timedelta(days=60)
        docs = [
            _make_doc(f"doc-{i}", f"Debugging tip {i}", category="debugging", created_at=old_time)
            for i in range(4)
        ]

        doc_store.get_all_documents = AsyncMock(return_value=docs)
        doc_store.delete_document = AsyncMock(return_value=True)

        job = SummarizationJob(memory)
        result = _run(job.run_async())

        assert result["categories_updated"] == 1
        mock_summarize.assert_called_once()
        memory.add_async.assert_awaited_once()

        # Verify the summary was stored with correct metadata
        call_kwargs = memory.add_async.call_args[1]
        assert call_kwargs["category"] == "category-summary"
        assert "category:debugging" in call_kwargs["tags"]

    @patch("cems.llm.summarize_memories")
    def test_summarization_skips_small_categories(self, mock_summarize, mock_memory):
        """Categories with fewer than 3 docs are not summarized."""
        from cems.maintenance.summarization import SummarizationJob

        memory, doc_store, _ = mock_memory

        old_time = datetime.now(UTC) - timedelta(days=60)
        docs = [
            _make_doc("doc-1", "Tip 1", category="small-cat", created_at=old_time),
            _make_doc("doc-2", "Tip 2", category="small-cat", created_at=old_time),
        ]

        doc_store.get_all_documents = AsyncMock(return_value=docs)
        doc_store.delete_document = AsyncMock(return_value=True)

        job = SummarizationJob(memory)
        result = _run(job.run_async())

        assert result["categories_updated"] == 0
        mock_summarize.assert_not_called()

    def test_summarization_prunes_stale(self, mock_memory):
        """Soft-deletes documents older than stale_days config."""
        from cems.maintenance.summarization import SummarizationJob

        memory, doc_store, _ = mock_memory

        stale_time = datetime.now(UTC) - timedelta(days=120)
        recent_time = datetime.now(UTC) - timedelta(days=5)

        docs = [
            _make_doc("stale-1", "Old content", updated_at=stale_time, created_at=stale_time),
            _make_doc("stale-2", "Old content 2", updated_at=stale_time, created_at=stale_time),
            _make_doc("fresh-1", "New content", updated_at=recent_time, created_at=recent_time),
        ]

        doc_store.get_all_documents = AsyncMock(return_value=docs)
        doc_store.delete_document = AsyncMock(return_value=True)

        job = SummarizationJob(memory)
        result = _run(job.run_async())

        assert result["memories_pruned"] == 2
        assert doc_store.delete_document.await_count == 2


class TestReindexJob:
    """Tests for the async ReindexJob."""

    def test_reindex_handles_empty(self, mock_memory):
        """No docs returns zeros."""
        from cems.maintenance.reindex import ReindexJob

        memory, doc_store, _ = mock_memory
        doc_store.get_all_documents = AsyncMock(return_value=[])

        job = ReindexJob(memory)
        result = _run(job.run_async())

        assert result["memories_reindexed"] == 0
        assert result["memories_archived"] == 0
        assert result["total_memories"] == 0

    def test_reindex_refreshes_embeddings(self, mock_memory):
        """Re-indexes all docs by calling update_async on each."""
        from cems.maintenance.reindex import ReindexJob

        memory, doc_store, _ = mock_memory

        docs = [
            _make_doc("doc-1", "Content 1"),
            _make_doc("doc-2", "Content 2"),
            _make_doc("doc-3", "Content 3"),
        ]
        doc_store.get_all_documents = AsyncMock(return_value=docs)
        doc_store.delete_document = AsyncMock(return_value=True)

        job = ReindexJob(memory)
        result = _run(job.run_async())

        assert result["memories_reindexed"] == 3
        assert memory.update_async.await_count == 3

    def test_reindex_archives_dead(self, mock_memory):
        """Soft-deletes documents older than archive_days (180)."""
        from cems.maintenance.reindex import ReindexJob

        memory, doc_store, _ = mock_memory

        dead_time = datetime.now(UTC) - timedelta(days=200)
        recent_time = datetime.now(UTC) - timedelta(days=10)

        docs = [
            _make_doc("dead-1", "Very old", updated_at=dead_time, created_at=dead_time),
            _make_doc("dead-2", "Also old", updated_at=dead_time, created_at=dead_time),
            _make_doc("alive-1", "Recent", updated_at=recent_time, created_at=recent_time),
        ]

        doc_store.get_all_documents = AsyncMock(return_value=docs)
        doc_store.delete_document = AsyncMock(return_value=True)

        job = ReindexJob(memory)
        result = _run(job.run_async())

        assert result["memories_archived"] == 2
        # 3 updates (reindex) + 2 deletes (archive)
        assert doc_store.delete_document.await_count == 2

    def test_reindex_uses_user_id(self, mock_memory):
        """Passes user_id correctly to DocumentStore."""
        from cems.maintenance.reindex import ReindexJob

        memory, doc_store, _ = mock_memory
        doc_store.get_all_documents = AsyncMock(return_value=[])
        doc_store.delete_document = AsyncMock(return_value=True)

        job = ReindexJob(memory)
        _run(job.run_async())

        doc_store.get_all_documents.assert_awaited_once_with(TEST_UUID, limit=5000)


class TestSchedulerIntegration:
    """Test scheduler correctly dispatches to async jobs."""

    def test_run_now_consolidation(self, mock_memory):
        """run_now('consolidation') dispatches to ConsolidationJob.run_async."""
        from cems.scheduler import CEMSScheduler

        memory, doc_store, _ = mock_memory
        doc_store.get_recent_documents = AsyncMock(return_value=[])

        config = memory.config
        scheduler = CEMSScheduler(config)
        result = scheduler.run_now("consolidation", memory)

        assert "duplicates_merged" in result
        assert "conflicts_found" in result
        assert "memories_checked" in result

    def test_run_now_summarization(self, mock_memory):
        """run_now('summarization') dispatches to SummarizationJob.run_async."""
        from cems.scheduler import CEMSScheduler

        memory, doc_store, _ = mock_memory
        doc_store.get_all_documents = AsyncMock(return_value=[])

        config = memory.config
        scheduler = CEMSScheduler(config)
        result = scheduler.run_now("summarization", memory)

        assert "categories_updated" in result
        assert "memories_pruned" in result

    def test_run_now_reindex(self, mock_memory):
        """run_now('reindex') dispatches to ReindexJob.run_async."""
        from cems.scheduler import CEMSScheduler

        memory, doc_store, _ = mock_memory
        doc_store.get_all_documents = AsyncMock(return_value=[])

        config = memory.config
        scheduler = CEMSScheduler(config)
        result = scheduler.run_now("reindex", memory)

        assert "memories_reindexed" in result
        assert "memories_archived" in result

    def test_run_now_reflect(self, mock_memory):
        """run_now('reflect') dispatches to ObservationReflector.run_async."""
        from cems.scheduler import CEMSScheduler

        memory, doc_store, _ = mock_memory
        doc_store.get_documents_by_category = AsyncMock(return_value=[])

        config = memory.config
        scheduler = CEMSScheduler(config)
        result = scheduler.run_now("reflect", memory)

        assert "projects_processed" in result

    def test_run_now_invalid_job(self, mock_memory):
        """run_now with invalid job type raises ValueError."""
        from cems.scheduler import CEMSScheduler

        memory, _, _ = mock_memory
        scheduler = CEMSScheduler(memory.config)

        with pytest.raises(ValueError, match="Unknown job type"):
            scheduler.run_now("invalid_job", memory)
