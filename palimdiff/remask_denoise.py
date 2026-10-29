from __future__ import annotations

from dataclasses import dataclass

from palimdiff.canvas import MemoryCanvas
from palimdiff.evidence_delta import (
    EvidenceDeltaConfig,
    score_evidence_delta,
    scores_to_spans,
    select_remask_positions,
)
from palimdiff.types import DenoiseContext, Denoiser, StepMetrics


@dataclass(frozen=True)
class RemaskDenoiseConfig:
    evidence: EvidenceDeltaConfig = EvidenceDeltaConfig()
    denoise_steps: int = 8


def evidence_delta_remask_denoise(
    canvas: MemoryCanvas,
    denoiser: Denoiser,
    context: DenoiseContext,
    *,
    config: RemaskDenoiseConfig,
) -> StepMetrics:
    scores = score_evidence_delta(
        denoiser,
        canvas.token_ids,
        context,
        config=config.evidence,
    )
    remask_positions = select_remask_positions(
        scores,
        rho=config.evidence.rho,
        capacity=canvas.capacity,
        free_region=canvas.free_region,  # type: ignore[arg-type]
        rng_seed=config.evidence.rng_seed + 7,
    )
    canvas.apply_remask(remask_positions)
    new_ids, nfe = denoiser.remask_and_denoise(
        canvas.token_ids,
        remask_positions,
        context,
        denoise_steps=config.denoise_steps,
        rng_seed=config.evidence.rng_seed + 13,
    )
    canvas.replace_tokens(new_ids)
    return StepMetrics(
        chunk_index=-1,
        remasked_tokens=len(remask_positions),
        total_memory_tokens=canvas.capacity,
        denoised_tokens=len(remask_positions),
        nfe=nfe,
        remask_spans=scores_to_spans(remask_positions),
    )


def fixed_remask_denoise(
    canvas: MemoryCanvas,
    denoiser: Denoiser,
    context: DenoiseContext,
    remask_positions: set[int],
    *,
    denoise_steps: int,
    rng_seed: int,
) -> StepMetrics:
    canvas.apply_remask(remask_positions)
    new_ids, nfe = denoiser.remask_and_denoise(
        canvas.token_ids,
        remask_positions,
        context,
        denoise_steps=denoise_steps,
        rng_seed=rng_seed,
    )
    canvas.replace_tokens(new_ids)
    return StepMetrics(
        chunk_index=-1,
        remasked_tokens=len(remask_positions),
        total_memory_tokens=canvas.capacity,
        denoised_tokens=len(remask_positions),
        nfe=nfe,
        remask_spans=scores_to_spans(remask_positions),
    )
