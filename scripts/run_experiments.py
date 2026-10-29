#!/usr/bin/env python3
"""Run all arms on built datasets; writes results/*.json (real tiny-stand-in runs labeled)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from baselines.registry import get_baseline_runner
from palimdiff.agent_loop import MemoryAgentConfig, UpdateStrategy, run_agent_loop
from palimdiff.denoisers.base import get_denoiser
from palimdiff.evidence_delta import EvidenceDeltaConfig
from palimdiff.remask_denoise import RemaskDenoiseConfig


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_arm(strategy: UpdateStrategy, denoiser, record: dict, seed: int, rho: float) -> dict:
    chunks = record.get("chunks") or record.get("passages", [])
    question = record["question"]
    cfg = MemoryAgentConfig(
        memory_capacity=64,
        strategy=strategy,
        rng_seed=seed,
        remask_denoise=RemaskDenoiseConfig(
            evidence=EvidenceDeltaConfig(rho=rho, rng_seed=seed),
            denoise_steps=4,
        ),
    )
    if strategy == UpdateStrategy.EVIDENCE_DELTA:
        result = run_agent_loop(denoiser, question, chunks, config=cfg)
    else:
        cfg.baseline_fn = get_baseline_runner(strategy)
        result = run_agent_loop(denoiser, question, chunks, config=cfg)
    denoised = sum(m.denoised_tokens for m in result.step_metrics)
    remasked = sum(m.remasked_tokens for m in result.step_metrics)
    return {
        "strategy": strategy.value,
        "record_id": record.get("id"),
        "denoised_tokens_total": denoised,
        "remasked_tokens_total": remasked,
        "n_chunks": len(chunks),
        "answer": result.answer,
        "mock_eval": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--denoiser", default="tiny")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rho", type=float, default=0.25)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    denoiser = get_denoiser(args.denoiser)
    retention = load_jsonl(Path("data/retention_probe/mock.jsonl"))
    ruler = load_jsonl(Path("data/ruler_hqa/mock.jsonl"))

    arms = [
        UpdateStrategy.EVIDENCE_DELTA,
        UpdateStrategy.FULL_OVERWRITE,
        UpdateStrategy.AR_OVERWRITE,
        UpdateStrategy.APPEND_TRUNCATE,
        UpdateStrategy.CONFIDENCE,
        UpdateStrategy.RANDOM,
    ]

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "denoiser": args.denoiser,
        "note": "CPU TinyStandIn runs — not GPU benchmark numbers. mock_eval=true.",
        "seed": args.seed,
        "rho": args.rho,
        "retention_probe": [],
        "ruler_hqa_mock": [],
    }

    for rec in retention[:3]:
        for arm in arms:
            payload["retention_probe"].append(run_arm(arm, denoiser, rec, args.seed, args.rho))

    for rec in ruler[:3]:
        for arm in arms:
            payload["ruler_hqa_mock"].append(run_arm(arm, denoiser, rec, args.seed, args.rho))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f"cpu_tiny_seed{args.seed}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
