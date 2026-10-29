from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Span:
    """Half-open interval [start, end) over memory token indices."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span: {self.start}:{self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class DenoiseContext:
    """Conditioning for a memory update step."""

    question: str
    chunk: str
    previous_chunk: str | None = None


@dataclass
class TokenScore:
    index: int
    evidence_delta: float
    pll_old: float
    pll_new: float


@dataclass
class StepMetrics:
    chunk_index: int
    remasked_tokens: int
    total_memory_tokens: int
    denoised_tokens: int
    nfe: int
    remask_spans: list[Span] = field(default_factory=list)


class Denoiser(Protocol):
    """Masked diffusion LM interface (real models or TinyStandIn)."""

    mask_token_id: int

    def tokenize(self, text: str) -> list[int]: ...

    def detokenize(self, token_ids: list[int]) -> str: ...

    def pseudo_log_likelihood(
        self,
        token_ids: list[int],
        context: DenoiseContext,
        *,
        k_partitions: int,
        rng_seed: int,
    ) -> list[float]:
        """Per-position PLL estimates (higher = more consistent with context)."""

    def remask_and_denoise(
        self,
        memory_token_ids: list[int],
        remask_positions: set[int],
        context: DenoiseContext,
        *,
        denoise_steps: int,
        rng_seed: int,
    ) -> tuple[list[int], int]:
        """Returns updated memory and NFE count."""

    def confidence_scores(
        self,
        token_ids: list[int],
        context: DenoiseContext,
        *,
        rng_seed: int,
    ) -> list[float]:
        """Per-token confidence in [0, 1] for confidence-remask baseline."""
