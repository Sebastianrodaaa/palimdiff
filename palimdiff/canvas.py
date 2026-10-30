from __future__ import annotations

from dataclasses import dataclass, field

from palimdiff.types import Span


@dataclass
class MemoryCanvas:
    """
    Fixed-length memory buffer with span bookkeeping for remask / keep regions.

    Tokens are integer ids aligned with the active Denoiser vocabulary.
    """

    capacity: int
    token_ids: list[int]
    free_region: Span | None = None
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.token_ids) != self.capacity:
            raise ValueError(f"expected {self.capacity} tokens, got {len(self.token_ids)}")
        if self.free_region is None:
            # Reserve trailing 5% (at least 1 token) as scratch / free region.
            free_len = max(1, self.capacity // 20)
            self.free_region = Span(self.capacity - free_len, self.capacity)

    @classmethod
    def from_text(cls, denoiser, text: str, capacity: int) -> MemoryCanvas:
        ids = denoiser.tokenize(text)
        if len(ids) > capacity:
            ids = ids[:capacity]
        pad = denoiser.mask_token_id
        ids = ids + [pad] * (capacity - len(ids))
        return cls(capacity=capacity, token_ids=ids)

    def kept_spans(self, remask_positions: set[int]) -> list[Span]:
        """Maximal contiguous spans not in remask_positions (excluding free region policy)."""
        spans: list[Span] = []
        start: int | None = None
        for i in range(self.capacity):
            if i in remask_positions:
                if start is not None:
                    spans.append(Span(start, i))
                    start = None
            elif start is None:
                start = i
        if start is not None:
            spans.append(Span(start, self.capacity))
        return spans

    def apply_remask(self, remask_positions: set[int]) -> None:
        """Mark positions as masked in-place (for visualization / bookkeeping)."""
        self.history.append(
            {
                "remask_positions": sorted(remask_positions),
                "kept_spans": [(s.start, s.end) for s in self.kept_spans(remask_positions)],
            }
        )

    def replace_tokens(self, new_ids: list[int]) -> None:
        if len(new_ids) != self.capacity:
            raise ValueError("new_ids length must match capacity")
        self.token_ids = list(new_ids)
