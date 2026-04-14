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
    shown_count: int = 1,
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
        "shown_count": shown_count,
        "last_shown_at": None,
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
        doc_store.delete_document.assert_awaited_once_with("doc-2", hard=False, user_id=TEST_UUID)

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

    def test_full_sweep_uses_get_all_documents(self, mock_memory):
        """full_sweep=True uses get_all_documents instead of get_recent_documents."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, _ = mock_memory
        doc_store.get_all_documents = AsyncMock(return_value=[])
        doc_store.get_recent_documents = AsyncMock(return_value=[])

        job = ConsolidationJob(memory)
        result = _run(job.run_async(full_sweep=True))

        assert result["memories_checked"] == 0
        assert result["offset"] == 0
        doc_store.get_all_documents.assert_awaited_once()
        doc_store.get_recent_documents.assert_not_awaited()

    def test_full_sweep_with_limit_and_offset(self, mock_memory):
        """full_sweep respects limit and offset params."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, _ = mock_memory
        doc1 = _make_doc("doc-1", "Content A")
        doc_store.get_all_documents = AsyncMock(return_value=[doc1])

        job = ConsolidationJob(memory)
        result = _run(job.run_async(full_sweep=True, limit=200, offset=100))

        assert result["memories_checked"] == 1
        assert result["offset"] == 100
        doc_store.get_all_documents.assert_awaited_once_with(
            TEST_UUID, limit=200, offset=100
        )

    def test_nightly_mode_uses_get_recent_documents(self, mock_memory):
        """Default (nightly) mode uses get_recent_documents."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, _ = mock_memory
        doc_store.get_recent_documents = AsyncMock(return_value=[])
        doc_store.get_all_documents = AsyncMock(return_value=[])

        job = ConsolidationJob(memory)
        result = _run(job.run_async())

        doc_store.get_recent_documents.assert_awaited_once()
        doc_store.get_all_documents.assert_not_awaited()
        assert "offset" not in result


class TestSummarizationJob:
    """Tests for the async SummarizationJob (entity-aware: archive compiled, prune noise)."""

    def test_summarization_no_docs(self, mock_memory):
        """Returns zeros when no documents found."""
        from cems.maintenance.summarization import SummarizationJob

        memory, doc_store, _ = mock_memory
        doc_store.get_all_documents = AsyncMock(return_value=[])
        doc_store.get_compiled_sources = AsyncMock(return_value=[])

        job = SummarizationJob(memory)
        result = _run(job.run_async())

        assert result["archived_compiled"] == 0
        assert result["noise_pruned"] == 0

    def test_summarization_archives_compiled_sources(self, mock_memory):
        """Archives memories that have been compiled into entity pages."""
        from cems.maintenance.summarization import SummarizationJob

        memory, doc_store, _ = mock_memory

        compiled_docs = [
            _make_doc("doc-1", "Old compiled memory 1"),
            _make_doc("doc-2", "Old compiled memory 2"),
            _make_doc("doc-3", "Old compiled memory 3"),
        ]

        doc_store.get_all_documents = AsyncMock(return_value=[])
        doc_store.get_compiled_sources = AsyncMock(return_value=compiled_docs)
        doc_store.delete_document = AsyncMock(return_value=True)

        job = SummarizationJob(memory)
        result = _run(job.run_async())

        assert result["archived_compiled"] == 3
        assert doc_store.delete_document.await_count == 3
        # Verify soft-delete (not hard)
        for call in doc_store.delete_document.call_args_list:
            assert call[1].get("hard") is False

    def test_summarization_prunes_noisy_memories(self, mock_memory):
        """Prunes memories with >50% noise rate and >=5 signals."""
        from cems.maintenance.summarization import SummarizationJob

        memory, doc_store, _ = mock_memory

        noisy_doc = _make_doc("noisy-1", "Always irrelevant")
        noisy_doc["relevant_count"] = 1
        noisy_doc["noise_count"] = 5  # 83% noise rate

        clean_doc = _make_doc("clean-1", "Mostly relevant")
        clean_doc["relevant_count"] = 8
        clean_doc["noise_count"] = 2  # 20% noise rate

        doc_store.get_all_documents = AsyncMock(return_value=[noisy_doc, clean_doc])
        doc_store.get_compiled_sources = AsyncMock(return_value=[])
        doc_store.delete_document = AsyncMock(return_value=True)

        job = SummarizationJob(memory)
        result = _run(job.run_async())

        assert result["noise_pruned"] == 1
        doc_store.delete_document.assert_awaited_once_with(
            "noisy-1", hard=False, user_id=TEST_UUID
        )

    def test_summarization_skips_protected_categories_for_noise(self, mock_memory):
        """Protected categories are not pruned even if noisy."""
        from cems.maintenance.summarization import SummarizationJob

        memory, doc_store, _ = mock_memory

        protected_doc = _make_doc("gate-1", "Gate rule", category="gate-rules")
        protected_doc["relevant_count"] = 0
        protected_doc["noise_count"] = 10

        doc_store.get_all_documents = AsyncMock(return_value=[protected_doc])
        doc_store.get_compiled_sources = AsyncMock(return_value=[])
        doc_store.delete_document = AsyncMock(return_value=True)

        job = SummarizationJob(memory)
        result = _run(job.run_async())

        assert result["noise_pruned"] == 0


class TestOrphanAssignerJob:
    """Tests for the OrphanAssignerJob — LLM-based orphan assignment."""

    def test_orphan_assigner_no_entity_pages(self, mock_memory):
        """Returns early when no entity pages exist."""
        from cems.maintenance.orphan_assigner import OrphanAssignerJob

        memory, doc_store, _ = mock_memory
        pool = AsyncMock()
        pool.fetch = AsyncMock(return_value=[])
        doc_store._get_pool = AsyncMock(return_value=pool)

        job = OrphanAssignerJob(memory)
        result = _run(job.run_async())

        assert result["orphans_found"] == 0
        assert result["assigned"] == 0
        assert "no entity pages" in result.get("message", "")

    def test_orphan_assigner_no_orphans(self, mock_memory):
        """Returns early when no orphan memories found."""
        from cems.maintenance.orphan_assigner import OrphanAssignerJob

        memory, doc_store, _ = mock_memory
        pool = AsyncMock()
        # First call: entity pages (returns some)
        entity_row = MagicMock()
        entity_row.__getitem__ = lambda s, k: {
            "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "title": "Docker Setup",
            "summary": "How to set up Docker...",
        }[k]
        # Second call: orphans (returns none)
        pool.fetch = AsyncMock(side_effect=[[entity_row], []])
        doc_store._get_pool = AsyncMock(return_value=pool)

        job = OrphanAssignerJob(memory)
        result = _run(job.run_async())

        assert result["orphans_found"] == 0
        assert result["assigned"] == 0

    @patch("cems.llm.client.get_client")
    def test_orphan_assigner_assigns_matching_orphan(self, mock_get_client, mock_memory):
        """Assigns orphan to matching entity page via LLM."""
        from cems.maintenance.orphan_assigner import OrphanAssignerJob

        memory, doc_store, _ = mock_memory

        entity_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

        # Mock the LLM client
        mock_client = MagicMock()
        mock_client.complete.return_value = "aaaaaaaa"
        mock_get_client.return_value = mock_client

        # Mock _get_entity_index and _get_orphan_memories directly
        job = OrphanAssignerJob(memory)
        job._get_entity_index = AsyncMock(return_value=[
            {"id": entity_id, "title": "Docker Setup", "summary": "Docker..."},
        ])
        job._get_orphan_memories = AsyncMock(return_value=[
            {"id": "orphan-1", "content": "Docker build failed on Mac", "category": "general", "source_ref": None},
        ])
        doc_store.add_relations = AsyncMock(return_value=1)

        result = _run(job.run_async())

        assert result["orphans_found"] == 1
        assert result["assigned"] == 1
        # Both forward and reverse relations should be added
        assert doc_store.add_relations.await_count == 2
        # Verify 'assigned_to' type (not 'compiled_from') to prevent archival
        for call in doc_store.add_relations.call_args_list:
            rel = call[0][1][0]
            assert rel["relation_type"] == "assigned_to"

    @patch("cems.llm.client.get_client")
    def test_orphan_assigner_skips_no_match(self, mock_get_client, mock_memory):
        """Skips orphans that don't match any entity page."""
        from cems.maintenance.orphan_assigner import OrphanAssignerJob

        memory, doc_store, _ = mock_memory

        mock_client = MagicMock()
        mock_client.complete.return_value = "none"
        mock_get_client.return_value = mock_client

        job = OrphanAssignerJob(memory)
        job._get_entity_index = AsyncMock(return_value=[
            {"id": "aaaaaaaa-1111-2222-3333-444444444444", "title": "Docker Setup", "summary": "Docker..."},
        ])
        job._get_orphan_memories = AsyncMock(return_value=[
            {"id": "orphan-1", "content": "Unrelated cooking recipe", "category": "general", "source_ref": None},
        ])

        result = _run(job.run_async())

        assert result["orphans_found"] == 1
        assert result["assigned"] == 0
        assert result["skipped"] == 1
        doc_store.add_relations.assert_not_awaited()

class TestCompilationJob:
    """Tests for CompilationJob — entity page synthesis from clusters."""

    def test_compilation_excludes_entity_pages_from_clusters(self, mock_memory):
        """Entity-page docs should not appear in discovered clusters."""
        from cems.maintenance.compilation import CompilationJob

        memory, doc_store, embedder = mock_memory

        # Mock pool with edges that include an entity-page source
        pool = AsyncMock()
        # Return edges where source is entity-page (should be excluded by query)
        pool.fetch = AsyncMock(return_value=[])  # No 'similar' edges
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=pool)
        pool.acquire.return_value.__aexit__ = AsyncMock()
        doc_store._get_pool = AsyncMock(return_value=pool)

        job = CompilationJob(memory)
        result = _run(job.run_async(limit=5))

        assert result["clusters_found"] == 0
        assert result["pages_created"] == 0

    def test_compilation_empty_returns_zeros(self, mock_memory):
        """No clusters means no pages."""
        from cems.maintenance.compilation import CompilationJob

        memory, doc_store, _ = mock_memory

        pool = AsyncMock()
        pool.fetch = AsyncMock(return_value=[])
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=pool)
        pool.acquire.return_value.__aexit__ = AsyncMock()
        doc_store._get_pool = AsyncMock(return_value=pool)

        job = CompilationJob(memory)
        result = _run(job.run_async())

        assert result["clusters_found"] == 0
        assert result["pages_created"] == 0
        assert result["pages_updated"] == 0


class TestCompilationDedup:
    """Tests for CompilationJob title-based dedup and merge logic."""

    @patch("cems.maintenance.compilation.CompilationJob._synthesize_entity")
    def test_title_dedup_skips_when_existing_entity_matches(self, mock_synth, mock_memory):
        """When a cluster's synthesized title matches an existing entity page,
        and no new sources exist, return 'skipped' instead of creating a dupe."""
        from cems.maintenance.compilation import CompilationJob

        memory, doc_store, embedder = mock_memory
        mock_synth.return_value = "# Fiscal Printer Integration\n\nOverview..."

        # Embedder returns identical vectors for title match
        embedder.embed_batch = AsyncMock(return_value=[[1.0, 0.0, 0.0]])

        # search_chunks returns a high-score match
        doc_store.search_chunks = AsyncMock(return_value=[
            {"document_id": "existing-entity-1", "score": 0.90, "content": "..."},
        ])

        # Existing entity already has all the same sources
        doc_store.get_related_documents = AsyncMock(return_value=[
            {"id": "doc-a"}, {"id": "doc-b"},
        ])

        doc_store.get_documents_by_tag = AsyncMock(return_value=[])

        cluster_docs = [
            _make_doc("doc-a", "Fiscal printer setup"),
            _make_doc("doc-b", "Z-Report reprint flow"),
        ]

        job = CompilationJob(memory)
        result = _run(job._compile_cluster(doc_store, TEST_UUID, cluster_docs, force=False))

        assert result == "skipped"
        # Should NOT have called add_async (no new page created)
        memory.add_async.assert_not_awaited()

    @patch("cems.maintenance.compilation.CompilationJob._synthesize_entity")
    def test_title_dedup_updates_when_new_sources(self, mock_synth, mock_memory):
        """When title matches an existing entity but cluster has new sources,
        update the existing entity page."""
        from cems.maintenance.compilation import CompilationJob

        memory, doc_store, embedder = mock_memory
        mock_synth.return_value = "# Fiscal Printer Integration\n\nOverview with new info..."

        embedder.embed_batch = AsyncMock(return_value=[[1.0, 0.0, 0.0]])

        doc_store.search_chunks = AsyncMock(return_value=[
            {"document_id": "existing-entity-1", "score": 0.90, "content": "..."},
        ])

        # Existing entity has only doc-a, but cluster has doc-a + doc-c (new)
        doc_store.get_related_documents = AsyncMock(return_value=[
            {"id": "doc-a"},
        ])

        doc_store.get_documents_by_tag = AsyncMock(return_value=[])
        doc_store.add_relations = AsyncMock(return_value=1)

        pool = AsyncMock()
        pool.execute = AsyncMock()
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=pool)
        pool.acquire.return_value.__aexit__ = AsyncMock()
        doc_store._get_pool = AsyncMock(return_value=pool)

        cluster_docs = [
            _make_doc("doc-a", "Fiscal printer setup"),
            _make_doc("doc-c", "New Z-Report edge case"),
        ]

        job = CompilationJob(memory)
        result = _run(job._compile_cluster(doc_store, TEST_UUID, cluster_docs, force=False))

        assert result == "updated"
        memory.update_async.assert_awaited_once()  # Updated existing
        memory.add_async.assert_not_awaited()  # Did NOT create new
        # Should have added relations for the new source (doc-c)
        assert doc_store.add_relations.await_count >= 2  # forward + reverse

    @patch("cems.maintenance.compilation.CompilationJob._synthesize_entity")
    def test_content_fallback_dedup(self, mock_synth, mock_memory):
        """When title doesn't match but content does (>0.78), dedup via content."""
        from cems.maintenance.compilation import CompilationJob

        memory, doc_store, embedder = mock_memory
        mock_synth.return_value = "# Docker Build Troubleshooting\n\nContent..."

        # Title embedding returns low score, content embedding returns high score
        call_count = 0
        async def fake_embed(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # Title
                return [[1.0, 0.0, 0.0]]
            else:  # Content
                return [[0.0, 1.0, 0.0]]

        embedder.embed_batch = fake_embed

        # Title search: no match above 0.80
        # Content search: match above 0.78
        search_call_count = 0
        async def fake_search(**kwargs):
            nonlocal search_call_count
            search_call_count += 1
            if search_call_count == 1:  # Title search
                return [{"document_id": "ent-1", "score": 0.60, "content": "..."}]
            else:  # Content search
                return [{"document_id": "ent-2", "score": 0.82, "content": "..."}]

        doc_store.search_chunks = fake_search
        doc_store.get_related_documents = AsyncMock(return_value=[])
        doc_store.get_documents_by_tag = AsyncMock(return_value=[])
        doc_store.add_relations = AsyncMock(return_value=1)

        pool = AsyncMock()
        pool.execute = AsyncMock()
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=pool)
        pool.acquire.return_value.__aexit__ = AsyncMock()
        doc_store._get_pool = AsyncMock(return_value=pool)

        cluster_docs = [_make_doc("doc-x", "Docker build failed")]

        job = CompilationJob(memory)
        result = _run(job._compile_cluster(doc_store, TEST_UUID, cluster_docs, force=False))

        assert result == "updated"
        memory.update_async.assert_awaited_once()

    def test_merge_duplicate_entities(self, mock_memory):
        """_merge_duplicate_entities soft-deletes lower-shown duplicates."""
        from cems.maintenance.compilation import CompilationJob

        memory, doc_store, embedder = mock_memory

        # Two entity pages with very similar titles
        entities = [
            {**_make_doc("ent-1", "Fiscal Printer Integration", category="entity-page", shown_count=10),
             "title": "Fiscal Printer Integration"},
            {**_make_doc("ent-2", "Fiscal Printer Integration in POS", category="entity-page", shown_count=3),
             "title": "Fiscal Printer Integration in POS"},
        ]

        doc_store.get_documents_by_category = AsyncMock(return_value=entities)

        # Embeddings: nearly identical vectors
        embedder.embed_batch = AsyncMock(return_value=[
            [0.9, 0.1, 0.0], [0.88, 0.12, 0.0],
        ])

        doc_store.get_related_documents = AsyncMock(return_value=[])
        doc_store.add_relations = AsyncMock(return_value=1)
        doc_store.delete_document = AsyncMock(return_value=True)

        job = CompilationJob(memory)
        merged = _run(job._merge_duplicate_entities(doc_store, TEST_UUID, limit=10))

        # Cosine of [0.9,0.1,0] and [0.88,0.12,0] = 0.9998... > 0.80
        assert merged == 1
        # Should soft-delete ent-2 (lower shown_count)
        doc_store.delete_document.assert_awaited_once_with("ent-2", hard=False, user_id=TEST_UUID)

    def test_merge_skips_dissimilar_entities(self, mock_memory):
        """Entities with different topics should not be merged."""
        from cems.maintenance.compilation import CompilationJob

        memory, doc_store, embedder = mock_memory

        entities = [
            {**_make_doc("ent-1", "Docker troubleshooting", category="entity-page"),
             "title": "Docker troubleshooting"},
            {**_make_doc("ent-2", "Pay rate guardrails", category="entity-page"),
             "title": "Pay rate guardrails"},
        ]

        doc_store.get_documents_by_category = AsyncMock(return_value=entities)

        # Embeddings: very different vectors
        embedder.embed_batch = AsyncMock(return_value=[
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        ])

        job = CompilationJob(memory)
        merged = _run(job._merge_duplicate_entities(doc_store, TEST_UUID, limit=10))

        assert merged == 0
        doc_store.delete_document.assert_not_awaited()


class TestLintDuplicateEntities:
    """Tests for LintJob duplicate entity page detection."""

    def test_lint_detects_duplicate_entities(self, mock_memory):
        """LintJob detects entity pages with similar titles."""
        from cems.maintenance.lint import LintJob

        memory, doc_store, embedder = mock_memory

        entities = [
            {**_make_doc("ent-1", "Fiscal Printer Integration", category="entity-page"),
             "title": "Fiscal Printer Integration"},
            {**_make_doc("ent-2", "Fiscal Printer Integration in POS", category="entity-page"),
             "title": "Fiscal Printer Integration in POS"},
        ]

        doc_store.get_documents_by_category = AsyncMock(return_value=entities)
        embedder.embed_batch = AsyncMock(return_value=[
            [0.9, 0.1, 0.0], [0.88, 0.12, 0.0],
        ])

        job = LintJob(memory)
        dupes = _run(job._detect_duplicate_entities(doc_store, TEST_UUID))

        assert len(dupes) == 1
        assert dupes[0]["entity_a"] == "ent-1"
        assert dupes[0]["entity_b"] == "ent-2"
        assert dupes[0]["similarity"] > 0.80

    def test_lint_no_duplicates_when_different(self, mock_memory):
        """No duplicates detected when entity titles are different."""
        from cems.maintenance.lint import LintJob

        memory, doc_store, embedder = mock_memory

        entities = [
            {**_make_doc("ent-1", "Docker setup", category="entity-page"),
             "title": "Docker setup"},
            {**_make_doc("ent-2", "Pay rate guardrails", category="entity-page"),
             "title": "Pay rate guardrails"},
        ]

        doc_store.get_documents_by_category = AsyncMock(return_value=entities)
        embedder.embed_batch = AsyncMock(return_value=[
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        ])

        job = LintJob(memory)
        dupes = _run(job._detect_duplicate_entities(doc_store, TEST_UUID))

        assert len(dupes) == 0


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

    @patch("cems.chunking.chunk_document")
    def test_reindex_refreshes_embeddings(self, mock_chunk, mock_memory):
        """Re-indexes stale docs (>7 days) via refresh_chunks (no updated_at bump)."""
        from cems.maintenance.reindex import ReindexJob

        memory, doc_store, embedder = mock_memory

        # Mock chunking and embedding
        mock_chunk_obj = MagicMock(seq=0, pos=0, content="c", tokens=1, bytes=1)
        mock_chunk.return_value = [mock_chunk_obj]
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 10])
        doc_store.refresh_chunks = AsyncMock(return_value=True)

        # Docs must be >7 days old to be re-indexed (fresh ones are skipped)
        old_time = datetime.now(UTC) - timedelta(days=30)
        docs = [
            _make_doc("doc-1", "Content 1", updated_at=old_time, created_at=old_time),
            _make_doc("doc-2", "Content 2", updated_at=old_time, created_at=old_time),
            _make_doc("doc-3", "Content 3", updated_at=old_time, created_at=old_time),
        ]
        doc_store.get_all_documents = AsyncMock(return_value=docs)
        doc_store.delete_document = AsyncMock(return_value=True)

        job = ReindexJob(memory)
        result = _run(job.run_async())

        assert result["memories_reindexed"] == 3
        assert doc_store.refresh_chunks.await_count == 3
        # update_async should NOT be called (would bump updated_at)
        memory.update_async.assert_not_awaited()

    @patch("cems.chunking.chunk_document")
    def test_reindex_archives_dead(self, mock_chunk, mock_memory):
        """Soft-deletes documents older than archive_days (60) by created_at."""
        from cems.maintenance.reindex import ReindexJob

        memory, doc_store, embedder = mock_memory

        # Mock chunking/embedding for the reindex portion
        mock_chunk_obj = MagicMock(seq=0, pos=0, content="c", tokens=1, bytes=1)
        mock_chunk.return_value = [mock_chunk_obj]
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 10])
        doc_store.refresh_chunks = AsyncMock(return_value=True)

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
        assert doc_store.delete_document.await_count == 2

    def test_reindex_uses_user_id(self, mock_memory):
        """Passes user_id correctly to DocumentStore."""
        from cems.maintenance.reindex import ReindexJob

        memory, doc_store, _ = mock_memory
        doc_store.get_all_documents = AsyncMock(return_value=[])
        doc_store.delete_document = AsyncMock(return_value=True)

        job = ReindexJob(memory)
        _run(job.run_async())

        doc_store.get_all_documents.assert_awaited_once_with(TEST_UUID, limit=5000, order="asc")


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
        doc_store.get_compiled_sources = AsyncMock(return_value=[])

        config = memory.config
        scheduler = CEMSScheduler(config)
        result = scheduler.run_now("summarization", memory)

        assert "archived_compiled" in result
        assert "noise_pruned" in result

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


class TestEntityPageProtection:
    """Entity pages must never be merged, deleted, or condensed by maintenance."""

    def test_consolidation_skips_entity_page_as_source(self, mock_memory):
        """ConsolidationJob skips entity-page docs when iterating source docs."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, _ = mock_memory
        entity_doc = _make_doc("entity-1", "# Wiki Page\nLong content here...")
        entity_doc["category"] = "entity-page"

        doc_store.get_recent_documents = AsyncMock(return_value=[entity_doc])
        memory._async_embedder = AsyncMock()
        memory._async_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])

        result = _run(ConsolidationJob(memory).run_async())

        # Entity page should be skipped — no search_chunks calls
        doc_store.search_chunks.assert_not_awaited()
        assert result["duplicates_merged"] == 0

    def test_consolidation_skips_entity_page_as_merge_target(self, mock_memory):
        """ConsolidationJob never deletes an entity-page found as merge candidate."""
        from cems.maintenance.consolidation import ConsolidationJob

        memory, doc_store, _ = mock_memory

        normal_doc = _make_doc("normal-1", "Some content about deployment")
        entity_doc = _make_doc("entity-1", "# Deployment Guide\nDetailed wiki...")
        entity_doc["category"] = "entity-page"

        doc_store.get_recent_documents = AsyncMock(return_value=[normal_doc])
        memory._async_embedder = AsyncMock()
        memory._async_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])

        # search_chunks returns the entity page as a near-duplicate
        doc_store.search_chunks = AsyncMock(return_value=[
            {"document_id": "entity-1", "score": 0.99},
        ])
        doc_store.get_document = AsyncMock(return_value=entity_doc)

        result = _run(ConsolidationJob(memory).run_async())

        # Entity page should NOT be deleted despite 0.99 similarity
        doc_store.delete_document.assert_not_awaited()
        assert result["duplicates_merged"] == 0
