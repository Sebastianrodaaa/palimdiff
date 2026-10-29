#!/usr/bin/env python3
"""Synthetic retention probe: facts planted in early chunks, queried at end."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_probe(out_path: Path, seed: int, n: int) -> None:
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i in range(n):
            secret = f"PLANT_{rng.randint(1000, 9999)}"
            early_chunk = f"The secret code is {secret}."
            filler = [f"noise_{rng.randint(0, 999)}" for _ in range(6)]
            record = {
                "id": f"retention_{i}",
                "question": "What is the secret code?",
                "answer": secret,
                "chunks": [early_chunk, *filler],
                "label": "MOCK_SYNTHETIC",
            }
            f.write(json.dumps(record) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/retention_probe/mock.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num", type=int, default=10)
    args = p.parse_args()
    build_probe(args.out, args.seed, args.num)
    print(f"Wrote retention probe to {args.out} (MOCK_SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
