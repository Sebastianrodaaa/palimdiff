from __future__ import annotations

from baselines.append_truncate import run_append_truncate
from baselines.ar_overwrite import run_ar_overwrite
from baselines.confidence import run_confidence_remask
from baselines.full_overwrite import run_full_overwrite
from baselines.random_remask import run_random_remask
from palimdiff.agent_loop import UpdateStrategy


def get_baseline_runner(strategy: UpdateStrategy):
    mapping = {
        UpdateStrategy.AR_OVERWRITE: run_ar_overwrite,
        UpdateStrategy.FULL_OVERWRITE: run_full_overwrite,
        UpdateStrategy.APPEND_TRUNCATE: run_append_truncate,
        UpdateStrategy.CONFIDENCE: run_confidence_remask,
        UpdateStrategy.RANDOM: run_random_remask,
    }
    if strategy not in mapping:
        raise KeyError(strategy)
    return mapping[strategy]
