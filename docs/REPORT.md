# palimdiff — research report (v0.1)

## Motivation

Long-horizon memory agents such as **MemAgent** ([2507.02259](https://arxiv.org/abs/2507.02259)) maintain a **fixed-size text memory** and **rewrite it entirely** with an autoregressive model after each document chunk. That design keeps context bounded but pays a **full-sequence rewrite per step** and can **erase early facts** when later chunks dominate the rewrite.

**Masked diffusion language models (dLLMs)** can update **only a subset** of token positions via remask-and-denoise. **palimdiff** asks whether a **chunk-conditioned evidence signal** can choose *which* memory tokens to remask so that contradictory or stale regions are rewritten while stable facts stay untouched.

## Method

1. **Memory canvas** — fixed-length token buffer with span bookkeeping and a reserved **free region** (default trailing ~5% of slots).
2. **Evidence delta** — for each memory position, estimate pseudo-log-likelihood (PLL) under the denoiser with **K random mask partitions**, conditioning on `(question, new chunk)` vs `(question, previous chunk)`. **Evidence delta** = PLL(new) − PLL(old). More negative delta → stronger candidate for remask.
3. **Remask policy (v1)** — remask top-**ρ** fraction by evidence delta (plus free region).
4. **Partial denoise** — re-denoise only remasked positions conditioned on question, new chunk, and kept memory.

**v2 (stretch, not implemented here):** GRPO-trained remask head on frozen denoiser hidden states with reward = answer correctness − λ × remasked fraction.

## Hypothesis and falsification (from brief)

**Hypothesis:** With 512-token memory and 32K–128K token documents, evidence-delta in-place editing beats dLLM **full overwrite** by ≥ **5 points absolute** on multi-hop long-document QA while denoising ≥ **50% fewer** memory tokens per chunk, and retains more early-planted facts on a **retention probe**.

**Falsified if:** Gain &lt; 2 points or within 3-seed variance, **or** **random remask at the same ρ** matches evidence-delta.

## Related work

See [RELATED_WORK.md](./RELATED_WORK.md). No prior work implements the full evidence-delta + persistent text memory canvas recipe; closest baselines are MemAgent (full AR overwrite), RemeDi (in-generation confidence remask), and Plan-and-Patch (failure-localized plan repair).

## Experimental setup (planned v1)

| Component | Planned setting |
|-----------|-----------------|
| Denoisers | LLaDA-8B-Instruct, Dream-v0-Instruct-7B (frozen) |
| AR baseline | Qwen2.5-7B-Instruct, MemAgent-style overwrite |
| Data | RULER-HQA-style HotpotQA + distractors (8K / 32K / 128K); synthetic retention probe |
| Arms | AR overwrite, dLLM full overwrite, append+truncate, confidence remask, random remask @ ρ, evidence-delta @ ρ ∈ {0.1, 0.25, 0.5} |
| Seeds | 3 |

**Licenses:** MIT for this repo; model/dataset cards must be checked before publishing checkpoints or derived data (see RELATED_WORK.md table).

## What was actually run (this repository state)

| Run | Environment | Label |
|-----|-------------|-------|
| `make repro` with **TinyStandInDenoiser** | CPU, no GPU | Real code path, **not** benchmark-quality |
| GPU experiments (LLaDA / Dream, RULER-HQA at scale) | **Not run** | **Pending** |

Outputs: `results/cpu_tiny_seed42.json` after `make repro` (if executed). All entries include `"mock_eval": true` for the bundled tiny runs. **No QA accuracy numbers from HotpotQA are reported** until GPU runs complete.

## Results

**GPU benchmark results: pending.**

Reproduce the CPU smoke pipeline:

```bash
make repro
```

Generate README table from JSON only:

```bash
python scripts/generate_readme_results.py
```

For full v1 experiments (when GPU available):

```bash
pip install -e ".[dev]"  # plus torch/transformers per model cards
python scripts/build_ruler_hqa.py --use-hotpot --out data/ruler_hqa/hotpot.jsonl
python scripts/run_experiments.py --denoiser llada --seed 42 --rho 0.25
python scripts/run_experiments.py --denoiser dream --seed 42 --rho 0.25
```

## Limitations

- **LLaDA / Dream backends** in this release delegate to **TinyStandIn** until Hugging Face weights and inference loops are wired (see `palimdiff/denoisers/`).
- **Evidence delta** via PLL is noisy at small K; cost scales with K × memory length.
- **Rule-based ρ** may be dominated by random remask at matched ρ if the signal is weak — the hypothesis explicitly tests this.

## Next steps

1. Implement real PLL + remask-denoise for LLaDA and Dream on GPU.
2. Run 3-seed RULER-HQA and retention probe at 512-token memory.
3. Add v2 GRPO remask head (`palimdiff/policy.py`, training script).
4. Compare wall-clock and NFEs vs MemAgent-style AR overwrite at matched accuracy.
