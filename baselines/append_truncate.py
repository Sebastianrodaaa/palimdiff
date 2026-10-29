from __future__ import annotations

from palimdiff.agent_loop import AgentRunResult, MemoryAgentConfig
from palimdiff.canvas import MemoryCanvas
from palimdiff.types import DenoiseContext, Denoiser, StepMetrics


def run_append_truncate(
    denoiser: Denoiser,
    question: str,
    chunks: list[str],
    config: MemoryAgentConfig,
) -> AgentRunResult:
    """Append new chunk tokens then truncate to capacity (no denoise)."""
    buffer: list[int] = []
    metrics: list[StepMetrics] = []
    for idx, chunk in enumerate(chunks):
        _ = DenoiseContext(question=question, chunk=chunk)
        buffer.extend(denoiser.tokenize(chunk))
        if len(buffer) > config.memory_capacity:
            buffer = buffer[-config.memory_capacity :]
        metrics.append(
            StepMetrics(
                chunk_index=idx,
                remasked_tokens=0,
                total_memory_tokens=config.memory_capacity,
                denoised_tokens=0,
                nfe=0,
            )
        )
    pad = denoiser.mask_token_id
    buffer = buffer + [pad] * (config.memory_capacity - len(buffer))
    ids = buffer[: config.memory_capacity]
    canvas = MemoryCanvas(capacity=config.memory_capacity, token_ids=ids)
    text = denoiser.detokenize(canvas.token_ids)
    return AgentRunResult(
        final_memory_text=text,
        step_metrics=metrics,
        answer=f"MOCK_ANSWER append_truncate mem_len={len(text)}",
    )
