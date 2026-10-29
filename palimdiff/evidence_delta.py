from __future__ import annotations

import math
from dataclasses import dataclass

from palimdiff.types import DenoiseContext, Denoiser, Span, TokenScore


@dataclass(frozen=True)
class EvidenceDeltaConfig:
    k_partitions: int = 4
    rho: float = 0.25
    min_span_len: int = 1
    rng_seed: int = 0


def score_evidence_delta(
    denoiser: Denoiser,
    memory_token_ids: list[int],
    context_new: DenoiseContext,
    *,
    config: EvidenceDeltaConfig,
) -> list[TokenScore]:
    """
    Evidence delta = PLL(context_new) - PLL(context_old).

    Large *negative* delta means the token fits worse when the new chunk is shown
    → candidate for remasking / rewrite.
    """
    ctx_old = DenoiseContext(
        question=context_new.question,
        chunk=context_new.previous_chunk or "",
        previous_chunk=None,
    )
    pll_new = denoiser.pseudo_log_likelihood(
        memory_token_ids,
        context_new,
        k_partitions=config.k_partitions,
        rng_seed=config.rng_seed,
    )
    pll_old = denoiser.pseudo_log_likelihood(
        memory_token_ids,
        ctx_old,
        k_partitions=config.k_partitions,
        rng_seed=config.rng_seed + 1,
    )
    scores: list[TokenScore] = []
    for i, (pn, po) in enumerate(zip(pll_new, pll_old, strict=True)):
        scores.append(TokenScore(index=i, evidence_delta=pn - po, pll_new=pn, pll_old=po))
    return scores


def select_remask_positions(
    scores: list[TokenScore],
    *,
    rho: float,
    capacity: int,
    free_region: Span,
    rng_seed: int,
) -> set[int]:
    """
    Top-rho fraction by evidence delta (most negative first), plus entire free region.
    """
    import random

    rng = random.Random(rng_seed)
    n_remask = max(1, int(math.floor(rho * capacity)))
    # Exclude free region from ranking pool (always remasked separately).
    pool = [s for s in scores if s.index not in range(free_region.start, free_region.end)]
    pool.sort(key=lambda s: s.evidence_delta)
    chosen = {s.index for s in pool[:n_remask]}
    for i in range(free_region.start, free_region.end):
        chosen.add(i)
    # Tie-break stability
    if len(chosen) < n_remask + free_region.length:
        rest = [s.index for s in pool if s.index not in chosen]
        rng.shuffle(rest)
        while len(chosen) < min(capacity, n_remask + free_region.length) and rest:
            chosen.add(rest.pop())
    return chosen


def scores_to_spans(positions: set[int]) -> list[Span]:
    if not positions:
        return []
    sorted_pos = sorted(positions)
    spans: list[Span] = []
    start = sorted_pos[0]
    prev = start
    for p in sorted_pos[1:]:
        if p == prev + 1:
            prev = p
        else:
            spans.append(Span(start, prev + 1))
            start = p
            prev = p
    spans.append(Span(start, prev + 1))
    return spans
