from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass

from palimdiff.types import DenoiseContext


@dataclass
class TinyStandInDenoiser:
    """
    Deterministic CPU stand-in mimicking mask-token PLL and partial denoise.

    Not a neural model — for tests and MOCK demos only.
    """

    mask_token_id: int = 0
    vocab_size: int = 256
    word_base: int = 10

    def tokenize(self, text: str) -> list[int]:
        words = re.findall(r"\w+", text.lower())
        ids: list[int] = []
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            ids.append(self.word_base + (h % (self.vocab_size - self.word_base)))
        return ids or [self.word_base]

    def detokenize(self, token_ids: list[int]) -> str:
        parts = []
        for tid in token_ids:
            if tid == self.mask_token_id:
                parts.append("[MASK]")
            else:
                parts.append(f"t{tid}")
        return " ".join(parts)

    def _context_vec(self, context: DenoiseContext) -> tuple[float, float]:
        blob = f"{context.question}|{context.chunk}"
        h = hashlib.sha256(blob.encode()).digest()
        return h[0] / 255.0, h[1] / 255.0

    def pseudo_log_likelihood(
        self,
        token_ids: list[int],
        context: DenoiseContext,
        *,
        k_partitions: int,
        rng_seed: int,
    ) -> list[float]:
        rng = random.Random(rng_seed)
        cx, cy = self._context_vec(context)
        scores: list[float] = []
        for i, tid in enumerate(token_ids):
            if tid == self.mask_token_id:
                scores.append(-2.0)
                continue
            # Token "fits" if its id aligns with context hash and position.
            align = math.sin((tid + i * 3) * 0.17 + cx * 6.28) * 0.5 + cy
            noise = rng.gauss(0, 0.05)
            scores.append(align + noise - (0.001 * k_partitions))
        return scores

    def remask_and_denoise(
        self,
        memory_token_ids: list[int],
        remask_positions: set[int],
        context: DenoiseContext,
        *,
        denoise_steps: int,
        rng_seed: int,
    ) -> tuple[list[int], int]:
        rng = random.Random(rng_seed)
        cx, cy = self._context_vec(context)
        out = list(memory_token_ids)
        nfe = 0
        chunk_ids = self.tokenize(context.chunk)
        for step in range(denoise_steps):
            nfe += len(remask_positions)
            for pos in sorted(remask_positions):
                if out[pos] == self.mask_token_id or step == denoise_steps - 1:
                    if chunk_ids:
                        pick = chunk_ids[(pos + step) % len(chunk_ids)]
                        out[pos] = pick
                    else:
                        out[pos] = self.word_base + int((cx * 100 + pos) % 50)
            # Partial convergence: unmask non-target positions unchanged
            rng.random()
        return out, nfe

    def confidence_scores(
        self,
        token_ids: list[int],
        context: DenoiseContext,
        *,
        rng_seed: int,
    ) -> list[float]:
        pll = self.pseudo_log_likelihood(token_ids, context, k_partitions=1, rng_seed=rng_seed)
        return [1.0 / (1.0 + math.exp(-p)) for p in pll]

    def generate_full(
        self,
        prompt: str,
        length: int,
        *,
        rng_seed: int,
    ) -> list[int]:
        """AR-style full rewrite baseline helper."""
        rng = random.Random(rng_seed)
        base = self.tokenize(prompt)
        out: list[int] = []
        for i in range(length):
            if i < len(base):
                out.append(base[i % len(base)])
            else:
                out.append(self.word_base + rng.randint(0, 40))
        return out[:length]
