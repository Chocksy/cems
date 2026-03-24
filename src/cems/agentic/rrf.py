"""Reciprocal Rank Fusion — shared between production search and eval."""

RRF_K = 60  # Standard RRF smoothing constant


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = RRF_K,
) -> list[str]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    For each item, score = sum(1 / (k + rank)) across all lists that contain it.

    Args:
        rankings: List of ranked ID lists (one per agent).
        k: Smoothing constant (standard=60).

    Returns:
        Merged list of IDs sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank_idx, item_id in enumerate(ranking):
            if item_id not in scores:
                scores[item_id] = 0.0
            scores[item_id] += 1.0 / (k + rank_idx + 1)

    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
