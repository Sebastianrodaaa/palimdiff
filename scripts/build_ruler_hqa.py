#!/usr/bin/env python3
"""Build RULER-HQA-style JSONL (synthetic fixtures, fixed seed).

Full HotpotQA requires Hugging Face ``datasets``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_synthetic_ruler(out_path: Path, seed: int, n: int) -> None:
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i in range(n):
            fact_a = f"entity_A_{i}"
            fact_b = f"entity_B_{i}"
            question = f"What connects {fact_a} and {fact_b}?"
            answer = f"link_{i}"
            distractors = [f"distractor_{rng.randint(0, 9999)}" for _ in range(8)]
            passages = [
                f"Passage about {fact_a}.",
                f"Passage about {fact_b}.",
                *distractors,
            ]
            rng.shuffle(passages)
            record = {
                "id": f"mock_ruler_{i}",
                "question": question,
                "answer": answer,
                "passages": passages,
                "label": "MOCK",
                "context_lengths": [8192, 32768, 131072],
            }
            f.write(json.dumps(record) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build RULER-HQA-style dataset (MOCK/synthetic by default).",
    )
    p.add_argument("--out", type=Path, default=Path("data/ruler_hqa/mock.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num", type=int, default=20)
    p.add_argument(
        "--use-hotpot",
        action="store_true",
        help="Attempt HotpotQA (requires datasets package).",
    )
    args = p.parse_args()
    if args.use_hotpot:
        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset("hotpot_qa", "distractor", split="validation[:10]")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w") as f:
                for row in ds:
                    f.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "question": row["question"],
                                "answer": row["answer"],
                                "passages": row["context"]["sentences"],
                                "label": "hotpot",
                            }
                        )
                        + "\n"
                    )
            print(f"Wrote HotpotQA subset to {args.out}")
            return
        except Exception as e:
            print(f"HotpotQA unavailable ({e}); falling back to MOCK synthetic.")
    build_synthetic_ruler(args.out, args.seed, args.num)
    print(f"Wrote MOCK RULER-HQA-style data to {args.out} (seed={args.seed})")


if __name__ == "__main__":
    main()
