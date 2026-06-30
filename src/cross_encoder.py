"""
Cross-encoder reranker (ONLINE, CPU-only, no network).

Loads 'cross-encoder/ms-marco-MiniLM-L-6-v2' from a LOCAL directory
(models/cross_encoder/) that must be populated offline by precompute.py.

A cross-encoder sees (JD, candidate) jointly with cross-attention,
detecting real career substance vs keyword stuffing - the #1 quality
gap vs naive bi-encoder cosine similarity.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Tuple

# Path where precompute.py caches the cross-encoder model
CE_MODEL_DIR = Path("models/cross_encoder")

_MODEL = None


def is_available() -> bool:
    """Return True if the cross-encoder model is cached locally."""
    return CE_MODEL_DIR.exists()


def _load_model():
    """Load cross-encoder from local cache (singleton). Fails loudly if absent."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not CE_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Cross-encoder model not found at '{CE_MODEL_DIR}'. "
            "Run precompute.py first to download it offline."
        )

    from sentence_transformers import CrossEncoder
    _MODEL = CrossEncoder(str(CE_MODEL_DIR), max_length=512)
    _MODEL.model.eval()
    return _MODEL


def rerank(
    jd_text: str,
    scored_candidates: List[Tuple],
    top_n: int = 200,
    ce_weight: float = 0.10,
) -> List[Tuple]:
    """Rerank the head of the shortlist with the cross-encoder.

    Args:
        jd_text: JD embedding text string.
        scored_candidates: List of (score, cid, profile, signal_scores), sorted desc.
        top_n: Number of head candidates to rerank (tail passes through unchanged).
        ce_weight: Weight to blend CE score into composite score.

    Returns:
        Reranked list - head reranked by CE+original blend, tail unchanged.
    """
    if not is_available():
        print(f"[cross_encoder] Model not found at {CE_MODEL_DIR}, skipping rerank.")
        return scored_candidates

    try:
        model = _load_model()
    except Exception as e:
        print(f"[cross_encoder] Failed to load model ({e}), skipping rerank.")
        return scored_candidates

    head = scored_candidates[:top_n]
    tail = scored_candidates[top_n:]

    # Build (JD, candidate_text) pairs - cap candidate text to stay within max_length
    pairs = []
    for _, _, profile, _ in head:
        candidate_text = profile.to_embedding_text()
        pairs.append((jd_text.strip()[:512], candidate_text[:700]))

    print(f"[cross_encoder] Reranking top {len(pairs)} candidates...")
    raw_scores = model.predict(pairs, show_progress_bar=False)

    # Sigmoid to normalize logits to [0, 1]
    ce_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]

    # Blend CE score into existing composite score
    reranked_head = []
    for i, (base_score, cid, profile, signal_scores) in enumerate(head):
        ce_score = ce_scores[i]
        blended_score = (1.0 - ce_weight) * base_score + ce_weight * ce_score
        signal_scores.cross_encoder_score = ce_score
        reranked_head.append((blended_score, cid, profile, signal_scores))

    # Re-sort the head by blended score, break ties by candidate_id ascending
    reranked_head.sort(key=lambda x: (-x[0], x[1]))

    ce_min = min(ce_scores)
    ce_max = max(ce_scores)
    print(f"[cross_encoder] Done. CE scores: {ce_min:.3f} - {ce_max:.3f}")
    return reranked_head + tail
