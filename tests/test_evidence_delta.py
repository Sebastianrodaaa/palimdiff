from palimdiff.denoisers.tiny import TinyStandInDenoiser
from palimdiff.evidence_delta import (
    EvidenceDeltaConfig,
    score_evidence_delta,
    select_remask_positions,
)
from palimdiff.types import DenoiseContext


def test_evidence_delta_scores_length():
    d = TinyStandInDenoiser()
    mem = d.tokenize("memory tokens here")
    mem = mem + [d.mask_token_id] * (16 - len(mem))
    ctx = DenoiseContext(question="q", chunk="new chunk", previous_chunk="old")
    cfg = EvidenceDeltaConfig(k_partitions=2, rng_seed=1)
    scores = score_evidence_delta(d, mem, ctx, config=cfg)
    assert len(scores) == 16


def test_select_remask_respects_rho():
    d = TinyStandInDenoiser()
    from palimdiff.canvas import MemoryCanvas

    c = MemoryCanvas.from_text(d, "x", capacity=20)
    scores = score_evidence_delta(
        d,
        c.token_ids,
        DenoiseContext("q", "c"),
        config=EvidenceDeltaConfig(rho=0.25, rng_seed=0),
    )
    pos = select_remask_positions(
        scores, rho=0.25, capacity=20, free_region=c.free_region, rng_seed=0
    )
    assert len(pos) >= 5
