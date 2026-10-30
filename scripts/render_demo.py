#!/usr/bin/env python3
"""Render MOCK demo GIF of memory canvas with remasked spans highlighted."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from palimdiff.agent_loop import MemoryAgentConfig
from palimdiff.denoisers.base import get_denoiser
from palimdiff.evidence_delta import EvidenceDeltaConfig
from palimdiff.remask_denoise import RemaskDenoiseConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/assets/demo_mock.gif"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    denoiser = get_denoiser("tiny")
    question = "What is the secret code?"
    chunks = [
        "The secret code is ALPHA.",
        "Unrelated weather report.",
        "More noise about sports.",
    ]
    cfg = MemoryAgentConfig(
        memory_capacity=48,
        rng_seed=args.seed,
        remask_denoise=RemaskDenoiseConfig(
            evidence=EvidenceDeltaConfig(rho=0.25, k_partitions=4, rng_seed=args.seed),
            denoise_steps=3,
        ),
    )

    # Run step-by-step to capture remask history
    from palimdiff.canvas import MemoryCanvas
    from palimdiff.remask_denoise import evidence_delta_remask_denoise
    from palimdiff.types import DenoiseContext

    canvas = MemoryCanvas.from_text(denoiser, "", cfg.memory_capacity)
    frames: list[np.ndarray] = []
    prev = None
    for idx, chunk in enumerate(chunks):
        ctx = DenoiseContext(question=question, chunk=chunk, previous_chunk=prev)
        step_cfg = RemaskDenoiseConfig(
            evidence=EvidenceDeltaConfig(rho=0.25, rng_seed=args.seed + idx),
            denoise_steps=3,
        )
        evidence_delta_remask_denoise(canvas, denoiser, ctx, config=step_cfg)
        hist = canvas.history[-1]["remask_positions"]
        grid_w = 12
        h = (cfg.memory_capacity + grid_w - 1) // grid_w
        fig, ax = plt.subplots(figsize=(6, 2.2), dpi=100)
        ax.set_xlim(0, grid_w)
        ax.set_ylim(0, h)
        ax.set_title(f"palimdiff memory canvas — chunk {idx + 1} (MOCK / TinyStandIn)")
        ax.axis("off")
        for i in range(cfg.memory_capacity):
            x = i % grid_w
            y = h - 1 - i // grid_w
            color = "#ff6b6b" if i in hist else "#4dabf7"
            ax.add_patch(Rectangle((x, y), 0.95, 0.95, facecolor=color, edgecolor="white"))
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)
        prev = chunk

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        args.out,
        save_all=True,
        append_images=pil_frames[1:],
        duration=900,
        loop=0,
    )
    print(f"Wrote MOCK demo GIF to {args.out}")


if __name__ == "__main__":
    main()
