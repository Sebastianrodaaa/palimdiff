from __future__ import annotations

from palimdiff.agent_loop import AgentRunResult, MemoryAgentConfig
from palimdiff.types import Denoiser, StepMetrics


def run_ar_overwrite(
    denoiser: Denoiser,
    question: str,
    chunks: list[str],
    config: MemoryAgentConfig,
) -> AgentRunResult:
    """MemAgent-style: regenerate entire memory string each chunk (AR stand-in)."""
    memory_ids: list[int] = []
    metrics: list[StepMetrics] = []
    for idx, chunk in enumerate(chunks):
        prompt = f"Q: {question}\nChunk: {chunk}\nMem:"
        memory_ids = denoiser.generate_full(  # type: ignore[attr-defined]
            prompt,
            config.memory_capacity,
            rng_seed=config.rng_seed + idx,
        )
        metrics.append(
            StepMetrics(
                chunk_index=idx,
                remasked_tokens=config.memory_capacity,
                total_memory_tokens=config.memory_capacity,
                denoised_tokens=config.memory_capacity,
                nfe=config.memory_capacity,
            )
        )
    text = denoiser.detokenize(memory_ids)
    return AgentRunResult(
        final_memory_text=text,
        step_metrics=metrics,
        answer=f"MOCK_ANSWER ar_overwrite mem_len={len(text)}",
    )
