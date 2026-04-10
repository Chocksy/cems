"""Retrieval operations for CEMSMemory (retrieve_for_inference pipeline)."""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any, Literal

from cems.models import SearchResult

if TYPE_CHECKING:
    from cems.memory.core import CEMSMemory

logger = logging.getLogger(__name__)

# Snippet limit for search results. Keeps context compact — LLMs can
# fetch the full document via GET /api/memory/get?id=<memory_id>.
_SNIPPET_CHARS = 500

# Session summary segments are joined with this separator.
# Strip it before snippeting to avoid noise in results.
_SEGMENT_SEP = "\n\n---\n\n"

# Pattern to detect content that starts mid-sentence
# (leading comma/period/semicolon, or lowercase after whitespace)
_MID_SENTENCE_RE = re.compile(r"^[\s,;.!?)}\]]+")


def _clean_content(content: str) -> str:
    """Clean content for snippet display.

    - Strips session summary segment separators (---)
    - Trims leading partial sentence fragments from mid-chunk starts
    """
    # Replace segment separators with single newline
    cleaned = content.replace(_SEGMENT_SEP, "\n\n")
    # Also handle bare --- lines (with varying whitespace)
    cleaned = re.sub(r"\n---\n", "\n", cleaned)

    # If chunk starts mid-sentence, skip to next sentence start
    match = _MID_SENTENCE_RE.match(cleaned)
    if match:
        cleaned = cleaned[match.end():]

    return cleaned.strip()


def _make_snippet(content: str) -> tuple[str, bool]:
    """Return (snippet, was_truncated)."""
    cleaned = _clean_content(content)
    if len(cleaned) <= _SNIPPET_CHARS:
        return cleaned, False
    cut = cleaned[:_SNIPPET_CHARS]
    for sep in (". ", ".\n", "\n\n", "\n"):
        pos = cut.rfind(sep)
        if pos > _SNIPPET_CHARS // 2:
            return cleaned[: pos + len(sep)].rstrip(), True
    return cut.rstrip() + "...", True


def _serialize_results(selected: list[SearchResult]) -> list[dict[str, Any]]:
    """Serialize SearchResult list with snippet truncation."""
    out: list[dict[str, Any]] = []
    for r in selected:
        snippet, truncated = _make_snippet(r.content)
        entry: dict[str, Any] = {
            "memory_id": r.memory_id,
            "content": snippet,
            "score": r.score,
            "scope": r.scope.value,
            "category": r.metadata.category if r.metadata else None,
            "source_ref": r.metadata.source_ref if r.metadata else None,
            "tags": r.metadata.tags if r.metadata else [],
            "created_at": str(r.metadata.created_at) if r.metadata and r.metadata.created_at else None,
        }
        if truncated:
            entry["truncated"] = True
            entry["full_length"] = len(r.content)
        if r.has_detailed:
            entry["has_detailed"] = True
        out.append(entry)
    return out


class RetrievalMixin:
    """Mixin class providing retrieval operations for CEMSMemory."""

    # NOTE: sync retrieve_for_inference was removed (296 LOC, zero callers).
    # The async version below is the only retrieval pipeline.
    # If sync is ever needed: _run_async(self.retrieve_for_inference_async(...))

    async def retrieve_for_inference_async(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        max_tokens: int = 2000,
        enable_query_synthesis: bool = True,
        enable_graph: bool = True,
        project: str | None = None,
        mode: Literal["auto", "vector", "hybrid"] = "auto",
        enable_hyde: bool = True,
        enable_decomposition: bool = True,
    ) -> dict[str, Any]:
        """Async version of retrieve_for_inference(). Use from HTTP server."""
        pipeline_start = time.perf_counter()
        from cems.retrieval import (
            _has_multi_topic_signals,
            apply_score_adjustments,
            assemble_context,
            assemble_context_diverse,
            decompose_query,
            deduplicate_results,
            extract_query_intent,
            format_memory_context,
            generate_hypothetical_memory,
            is_strong_lexical_signal,
            reciprocal_rank_fusion,
            route_to_strategy,
            synthesize_query,
        )

        logger.info(f"[RETRIEVAL] Starting async retrieve_for_inference: query='{query[:50]}...'")

        # Ensure async embedder is initialized
        await self._ensure_initialized_async()
        assert self._async_embedder is not None

        from cems.llm import get_client
        client = get_client()

        intent = None
        selected_mode = mode
        if mode == "auto" and client:
            intent = extract_query_intent(query, client)
            selected_mode = route_to_strategy(intent)
            logger.info(f"[RETRIEVAL] Auto mode selected: {selected_mode}")

        # Detect query types - each needs different handling
        from cems.retrieval import _is_temporal_query, _is_preference_query, _is_aggregation_query
        is_temporal = _is_temporal_query(query)
        is_preference = _is_preference_query(query)
        is_aggregation = _is_aggregation_query(query)

        if is_temporal:
            logger.info(f"[RETRIEVAL] Temporal query detected - will use query synthesis for decomposition")
        if is_preference:
            logger.info(f"[RETRIEVAL] Preference query detected - will use query synthesis to bridge semantic gap")
        if is_aggregation:
            logger.info(f"[RETRIEVAL] Aggregation query detected - will use larger candidate pool and diversity selection")

        # Stage 1.5: Multi-topic decomposition
        is_decomposed = False
        sub_queries: list[str] = []
        if (enable_decomposition and self.config.enable_query_decomposition
                and client and not (is_temporal or is_preference or is_aggregation)):
            if _has_multi_topic_signals(query):
                sub_queries = decompose_query(query, client, max_queries=self.config.max_decomposed_queries)
                if len(sub_queries) > 1:
                    is_decomposed = True
                    logger.info(f"[RETRIEVAL] Decomposed into {len(sub_queries)} sub-queries")

        # Stage 2.0: Profile Probe FIRST for preference queries (RAP approach)
        # We need profile_context before synthesis to provide dynamic examples
        #
        # NOTE: Adaptive probe approaches were tried but ALL HURT performance:
        # - v1 (no filter): 40% preference (down from 56.7% baseline)
        # - v2 (strict LLM filter): 50% preference
        # - v3 (lenient LLM filter): 46.7% preference
        #
        # Root cause: LLM filter calibration is difficult - either too strict
        # (removes good prefs) or too lenient (keeps bad prefs). The simple
        # fixed probe works best for now.
        profile_context: list[str] = []
        if is_preference and not is_decomposed:
            from cems.retrieval import extract_profile_context
            profile_probe_query = "I use I prefer my favorite I recently I took a class I really like"
            profile_results = await self._search_raw_async(profile_probe_query, scope, limit=5)
            if profile_results:
                profile_context = extract_profile_context([r.content for r in profile_results])
                if profile_context:
                    logger.info(f"[RETRIEVAL] Profile probe found context: {profile_context[:3]}...")

        # Stage 2.1: Query synthesis with strong-signal skip
        queries_to_search = [query]
        skip_expansion = False

        if is_decomposed:
            queries_to_search = [query] + sub_queries
            logger.info(f"[RETRIEVAL] Using decomposed queries: {len(queries_to_search)} total")
        else:
            # ALWAYS run synthesis for temporal/preference/aggregation queries (they need expansion)
            enable_preference = getattr(self.config, 'enable_preference_synthesis', True)
            force_synthesis = is_temporal or (is_preference and enable_preference) or is_aggregation
            should_synthesize = client and (enable_query_synthesis or force_synthesis) and (
                self.config.enable_query_synthesis or force_synthesis
            )
            if force_synthesis:
                query_type = 'temporal' if is_temporal else ('aggregation' if is_aggregation else 'preference')
                logger.info(f"[RETRIEVAL] Forcing synthesis for {query_type} query")
            if should_synthesize:
                # Probe BM25 to check signal strength before expanding
                lexical_probe = await self._search_lexical_raw_async(query, scope, limit=2)
                if lexical_probe:
                    top_score = lexical_probe[0].score
                    second_score = lexical_probe[1].score if len(lexical_probe) > 1 else 0.0
                    threshold = self.config.strong_signal_threshold
                    gap_threshold = self.config.strong_signal_gap
                    if is_strong_lexical_signal(top_score, second_score, threshold, gap_threshold):
                        gap = top_score - second_score
                        logger.info(
                            f"[RETRIEVAL] Strong signal detected (score={top_score:.3f}, "
                            f"gap={gap:.3f}), skipping query expansion"
                        )
                        skip_expansion = True

                if not skip_expansion:
                    # RAP: Pass profile_context as dynamic examples to synthesis
                    expanded = synthesize_query(query, client, is_preference=is_preference, profile_context=profile_context)
                    queries_to_search = [query] + expanded[:3]
                    logger.info(f"[RETRIEVAL] Query synthesis: {len(queries_to_search)} queries")

        # For preference queries, ALWAYS enable HyDE to bridge semantic gap
        should_hyde = enable_hyde and selected_mode == "hybrid" and client
        if is_preference and client:
            # Force HyDE for preference queries - critical for bridging semantic gap
            should_hyde = True
            logger.info(f"[RETRIEVAL] Forcing HyDE for preference query")
        if should_hyde:
            hypothetical = generate_hypothetical_memory(
                query, client, is_preference=is_preference, profile_context=profile_context
            )
            if hypothetical:
                queries_to_search.append(hypothetical)
                logger.info(f"[RETRIEVAL] HyDE generated")

        # OPTIMIZATION: Batch embed all queries in a single API call
        # This reduces N sequential embedding calls (~500ms each) to 1 batch call
        embed_start = time.perf_counter()
        logger.info(f"[RETRIEVAL] Batch embedding {len(queries_to_search)} queries")
        query_embeddings = await self._async_embedder.embed_batch(queries_to_search)
        embed_ms = (time.perf_counter() - embed_start) * 1000
        logger.info(f"[TIMING] Batch embedding complete: {embed_ms:.0f}ms for {len(queries_to_search)} queries")

        # Stage 4: Candidate retrieval using pre-computed embeddings
        # Track which lists are original vs expansion for RRF weights
        query_results: list[list[SearchResult]] = []
        list_weights: list[float] = []
        enable_lexical = self.config.enable_lexical_in_inference

        # For aggregation queries, use larger candidate pool to find more relevant memories
        candidates_limit = self.config.max_candidates_per_query
        if is_aggregation:
            candidates_limit = max(50, candidates_limit * 2)  # At least 50, or 2x default
            logger.info(f"[RETRIEVAL] Aggregation query: using larger candidate pool ({candidates_limit})")

        # Determine sub-query boundary for weight assignment
        _decomp_end = 1 + len(sub_queries) if is_decomposed else 1

        search_start = time.perf_counter()
        for i, (search_query, embedding) in enumerate(zip(queries_to_search, query_embeddings)):
            is_original = (i == 0)  # First query is the original
            if is_original:
                weight = self.config.rrf_original_weight
            elif is_decomposed and i < _decomp_end:
                weight = self.config.rrf_decomposition_weight
            else:
                weight = self.config.rrf_expansion_weight

            # Vector search (scores already 0-1)
            vector_results = await self._search_raw_async(
                search_query, scope, limit=candidates_limit,
                query_embedding=embedding,
            )
            query_results.append(vector_results)
            list_weights.append(weight)

            # Lexical search (BM25 scores need normalization)
            if enable_lexical:
                lexical_results = await self._search_lexical_raw_async(
                    search_query, scope, limit=50
                )
                # CRITICAL: Normalize BM25 scores to 0-1 (BM25 returns 0-5+)
                if lexical_results:
                    max_score = max(r.score for r in lexical_results)
                    if max_score > 0:
                        for r in lexical_results:
                            r.score = r.score / max_score
                query_results.append(lexical_results)
                list_weights.append(weight)
        search_ms = (time.perf_counter() - search_start) * 1000
        logger.info(f"[TIMING] DB search (vector+lexical): {search_ms:.0f}ms for {len(queries_to_search)} queries")

        # Log raw candidate counts per query
        for i, results in enumerate(query_results):
            if results:
                top_scores = [f"{r.score:.3f}" for r in results[:3]]
                logger.info(f"[RETRIEVAL] Query #{i}: {len(results)} results, top scores: {top_scores}")

        if enable_graph and query_results and query_results[0]:
            relation_results: list[SearchResult] = []
            for top_result in query_results[0][:5]:
                related = await self.get_related_memories_async(top_result.memory_id, limit=8)
                for rel in related:
                    metadata = await self.get_metadata_async(rel["id"])
                    if metadata:
                        base_score = rel.get("relation_similarity", 0.3) or 0.3
                        relation_results.append(
                            SearchResult(
                                memory_id=rel["id"],
                                content=rel.get("content", ""),
                                score=base_score,
                                scope=metadata.scope,
                                metadata=metadata,
                            )
                        )

            if relation_results:
                query_results.append(relation_results)
                # Relations get lower weight (0.5x)
                list_weights.append(0.5)

        if len(query_results) > 1:
            # Pass weights and top-rank bonus from config
            top_rank_bonus = (
                self.config.rrf_top_rank_bonus_r1,
                self.config.rrf_top_rank_bonus_r23,
            )
            candidates = reciprocal_rank_fusion(
                query_results,
                list_weights=list_weights,
                top_rank_bonus=top_rank_bonus,
            )
            logger.info(f"[RETRIEVAL] RRF fusion: {sum(len(r) for r in query_results)} -> {len(candidates)} results")
        else:
            candidates = query_results[0] if query_results else []

        candidates = deduplicate_results(candidates)

        threshold = self.config.relevance_threshold
        before_filter = len(candidates)
        candidates = [c for c in candidates if c.score >= threshold]
        logger.info(
            f"[RETRIEVAL] Relevance filter: threshold={threshold:.2f}, "
            f"{before_filter} -> {len(candidates)} candidates"
        )

        for candidate in candidates:
            candidate.score = apply_score_adjustments(
                candidate,
                project=project,
                config=self.config,
            )

        candidates.sort(key=lambda x: x.score, reverse=True)

        # Score-gap filter: drop results far below the top score (adaptive cutoff)
        if candidates and not is_aggregation:
            cutoff = candidates[0].score * self.config.score_gap_ratio
            before_gap = len(candidates)
            candidates = [c for i, c in enumerate(candidates) if i < 2 or c.score >= cutoff]
            if len(candidates) < before_gap:
                logger.info(f"[RETRIEVAL] Score-gap filter: {before_gap} -> {len(candidates)} (cutoff={cutoff:.3f})")

        total_candidates = sum(len(r) for r in query_results)
        filtered_count = len(candidates)

        # Use diverse assembly for aggregation queries to ensure session diversity
        # Aggregation queries need larger token budget to fit results from multiple sessions
        assembly_budget = max_tokens
        if is_aggregation:
            assembly_budget = max(max_tokens, 4000)  # At least 4000 tokens for aggregation
            logger.info(f"[RETRIEVAL] Aggregation query: increased token budget from {max_tokens} to {assembly_budget}")
            selected, tokens_used = assemble_context_diverse(candidates, assembly_budget)
        else:
            selected, tokens_used = assemble_context(candidates, assembly_budget)

        # Log final selection with source_refs
        assembly_type = "diverse" if is_aggregation else "standard"
        logger.info(f"[RETRIEVAL] Final ({assembly_type}) {len(selected)} results:")
        for i, r in enumerate(selected[:5]):
            src_ref = r.metadata.source_ref if r.metadata else "NONE"
            logger.info(f"  [{i}] score={r.score:.3f} src={src_ref} id={r.memory_id[:8]}...")

        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info(f"[TIMING] PIPELINE TOTAL: {pipeline_ms:.0f}ms | {filtered_count} candidates -> {len(selected)} selected, {tokens_used} tokens")

        return {
            "results": _serialize_results(selected),
            "tokens_used": tokens_used,
            "formatted_context": format_memory_context(selected),
            "queries_used": queries_to_search,
            "total_candidates": total_candidates,
            "filtered_count": filtered_count,
            "mode": selected_mode,
            "intent": intent,
        }
