"""End-to-end CPU smoke test on MOCK fixtures."""

import json
from pathlib import Path

from baselines.registry import get_baseline_runner
from palimdiff.agent_loop import MemoryAgentConfig, UpdateStrategy, run_agent_loop
from palimdiff.denoisers.tiny import TinyStandInDenoiser
from palimdiff.evidence_delta import EvidenceDeltaConfig
from palimdiff.remask_denoise import RemaskDenoiseConfig


def test_e2e_evidence_delta_mock_fixture():
    fixture = Path(__file__).parent / "mock_data" / "sample_retention.json"
    record = json.loads(fixture.read_text())
    assert record["label"] == "MOCK"

    denoiser = TinyStandInDenoiser()
    cfg = MemoryAgentConfig(
        memory_capacity=32,
        strategy=UpdateStrategy.EVIDENCE_DELTA,
        rng_seed=42,
        remask_denoise=RemaskDenoiseConfig(
            evidence=EvidenceDeltaConfig(rho=0.25, k_partitions=2, rng_seed=42),
            denoise_steps=2,
        ),
    )
    result = run_agent_loop(
        denoiser,
        record["question"],
        record["chunks"],
        config=cfg,
    )
    assert result.step_metrics
    assert result.step_metrics[0].denoised_tokens > 0
    assert "MOCK" in result.answer or "mem_len" in result.answer


def test_baseline_full_overwrite_runs():
    denoiser = TinyStandInDenoiser()
    cfg = MemoryAgentConfig(memory_capacity=24, rng_seed=1)
    cfg.baseline_fn = get_baseline_runner(UpdateStrategy.FULL_OVERWRITE)
    result = run_agent_loop(denoiser, "q?", ["c1", "c2"], config=cfg)
    assert len(result.step_metrics) == 2
    assert result.step_metrics[0].remasked_tokens == 24
