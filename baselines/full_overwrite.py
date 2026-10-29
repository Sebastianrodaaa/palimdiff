from __future__ import annotations

from palimdiff.agent_loop import AgentRunResult, MemoryAgentConfig
from palimdiff.canvas import MemoryCanvas
from palimdiff.remask_denoise import fixed_remask_denoise
from palimdiff.types import DenoiseContext, Denoiser, StepMetrics


def run_full_overwrite(
    denoiser: Denoiser,
    question: str,
    chunks: list[str],
    config: MemoryAgentConfig,
) -> AgentRunResult:
    canvas = MemoryCanvas.from_text(denoiser, "", config.memory_capacity)
    metrics: list[StepMetrics] = []
    prev: str | None = None
    all_positions = set(range(canvas.capacity))
    for idx, chunk in enumerate(chunks):
        ctx = DenoiseContext(question=question, chunk=chunk, previous_chunk=prev)
        m = fixed_remask_denoise(
            canvas,
            denoiser,
            ctx,
            all_positions,
            denoise_steps=config.remask_denoise.denoise_steps,
            rng_seed=config.rng_seed + idx,
        )
        m.chunk_index = idx
        metrics.append(m)
        prev = chunk
    text = denoiser.detokenize(canvas.token_ids)
    return AgentRunResult(
        final_memory_text=text,
        step_metrics=metrics,
        answer=f"MOCK_ANSWER full_overwrite mem_len={len(text)}",
    )
