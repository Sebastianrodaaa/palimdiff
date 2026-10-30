from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from palimdiff.canvas import MemoryCanvas
from palimdiff.evidence_delta import EvidenceDeltaConfig
from palimdiff.remask_denoise import RemaskDenoiseConfig, evidence_delta_remask_denoise
from palimdiff.types import DenoiseContext, Denoiser, StepMetrics


class UpdateStrategy(str, Enum):
    EVIDENCE_DELTA = "evidence_delta"
    FULL_OVERWRITE = "full_overwrite"
    APPEND_TRUNCATE = "append_truncate"
    CONFIDENCE = "confidence"
    RANDOM = "random"
    AR_OVERWRITE = "ar_overwrite"


@dataclass
class MemoryAgentConfig:
    memory_capacity: int = 64
    strategy: UpdateStrategy = UpdateStrategy.EVIDENCE_DELTA
    remask_denoise: RemaskDenoiseConfig = field(default_factory=RemaskDenoiseConfig)
    rng_seed: int = 42
    baseline_fn: Callable | None = None  # injected by baselines package


@dataclass
class AgentRunResult:
    final_memory_text: str
    step_metrics: list[StepMetrics]
    answer: str


def run_agent_loop(
    denoiser: Denoiser,
    question: str,
    chunks: list[str],
    *,
    config: MemoryAgentConfig,
    initial_memory: str = "",
) -> AgentRunResult:
    if config.baseline_fn is not None:
        return config.baseline_fn(denoiser, question, chunks, config)

    canvas = (
        MemoryCanvas.from_text(denoiser, initial_memory, config.memory_capacity)
        if initial_memory
        else MemoryCanvas.from_text(denoiser, "", config.memory_capacity)
    )
    metrics: list[StepMetrics] = []
    prev_chunk: str | None = None

    for idx, chunk in enumerate(chunks):
        ctx = DenoiseContext(question=question, chunk=chunk, previous_chunk=prev_chunk)
        ev = config.remask_denoise.evidence
        step_cfg = RemaskDenoiseConfig(
            evidence=EvidenceDeltaConfig(
                k_partitions=ev.k_partitions,
                rho=ev.rho,
                min_span_len=ev.min_span_len,
                rng_seed=config.rng_seed + idx * 101,
            ),
            denoise_steps=config.remask_denoise.denoise_steps,
        )
        m = evidence_delta_remask_denoise(canvas, denoiser, ctx, config=step_cfg)
        m.chunk_index = idx
        metrics.append(m)
        prev_chunk = chunk

    memory_text = denoiser.detokenize(canvas.token_ids)
    answer = _answer_from_memory(denoiser, question, memory_text, config.rng_seed)
    return AgentRunResult(final_memory_text=memory_text, step_metrics=metrics, answer=answer)


def _answer_from_memory(
    denoiser: Denoiser,
    question: str,
    memory_text: str,
    seed: int,
) -> str:
    """MOCK answer extraction for tiny stand-in (real runs use task eval)."""
    _ = denoiser
    return f"MOCK_ANSWER seed={seed} q={question[:32]!r} mem_len={len(memory_text)}"
