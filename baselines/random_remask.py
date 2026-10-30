from __future__ import annotations

import math
import random

from palimdiff.agent_loop import AgentRunResult, MemoryAgentConfig
from palimdiff.canvas import MemoryCanvas
from palimdiff.remask_denoise import fixed_remask_denoise
from palimdiff.types import DenoiseContext, Denoiser, StepMetrics


def run_random_remask(
    denoiser: Denoiser,
    question: str,
    chunks: list[str],
    config: MemoryAgentConfig,
) -> AgentRunResult:
    canvas = MemoryCanvas.from_text(denoiser, "", config.memory_capacity)
    metrics: list[StepMetrics] = []
    prev: str | None = None
    rho = config.remask_denoise.evidence.rho
    n_remask = max(1, int(math.floor(rho * canvas.capacity)))
    free = canvas.free_region
    assert free is not None
    rng = random.Random(config.rng_seed)

    for idx, chunk in enumerate(chunks):
        ctx = DenoiseContext(question=question, chunk=chunk, previous_chunk=prev)
        pool = [i for i in range(canvas.capacity) if i not in range(free.start, free.end)]
        rng.shuffle(pool)
        remask = set(pool[:n_remask])
        for i in range(free.start, free.end):
            remask.add(i)
        m = fixed_remask_denoise(
            canvas,
            denoiser,
            ctx,
            remask,
            denoise_steps=config.remask_denoise.denoise_steps,
            rng_seed=config.rng_seed + idx + 99,
        )
        m.chunk_index = idx
        metrics.append(m)
        prev = chunk

    text = denoiser.detokenize(canvas.token_ids)
    return AgentRunResult(
        final_memory_text=text,
        step_metrics=metrics,
        answer=f"MOCK_ANSWER random mem_len={len(text)}",
    )
