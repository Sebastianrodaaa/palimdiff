# Related work: in-place / partial memory editing with diffusion LMs for agents

**Search date:** 2026-10-29 (arXiv, Hugging Face Hub, GitHub code search).

## Summary

No published work or public repository implements **exactly** palimdiff’s recipe: a **fixed-length text memory canvas** updated across document chunks, where **remask regions are chosen by cross-chunk evidence delta** (pseudo-log-likelihood drop under random mask partitions when conditioning on the new chunk vs. the previous one), followed by **partial re-denoising** of only those spans. Closest lines of work use full memory overwrite (AR), latent/parametric memory inside denoising, plan repair on failure, or within-generation confidence remasking—none combine persistent agent text memory with evidence-conditioned partial dLLM edits.

**Narrow contribution (still novel):** evidence-delta scoring as a **rule-based, chunk-conditioned remask policy** for long-horizon agent memory, with baselines at matched remask fraction ρ and a retention probe; v2 (GRPO remask head) is explicitly future work.

---

## Papers checked (from research brief)

| Work | ID / link | Memory object | Partial edit? | Cross-step evidence for remask? | Relation to palimdiff |
|------|-----------|---------------|---------------|----------------------------------|------------------------|
| **MemAgent** | [2507.02259](https://arxiv.org/abs/2507.02259) | Fixed-length **text** memory | No — **full overwrite** each chunk (AR) | RL reward on final QA, not per-token evidence delta | Same agent loop shape; palimdiff swaps AR full rewrite for dLLM **selective** rewrite. |
| **LycheeMemory** | [2602.08382](https://arxiv.org/abs/2602.08382) | Plaintext working memory + compressed KV bank | Gate skips chunks; reasoner **updates** working memory (AR-style) | Relevance gate, not PLL delta on memory tokens | Long-context + RULER-HQA; **no diffusion** in-place token editing. Repo: [owoakuma/LycheeMemory](https://github.com/owoakuma/LycheeMemory). |
| **Plan-and-Patch** | [2610.10786](https://arxiv.org/abs/2610.10786) | **Plan** text (not persistent fact memory) | Yes — dLLM infill of repair region | Executor **failure** feedback, not new document chunk | Shows dLLM partial rewrite for agents; **not** streaming memory over long documents. |
| **MetaState** | [2603.01331](https://arxiv.org/abs/2603.01331) | **Latent** slots across denoising steps | N/A (inside one generation) | Timestep recurrence, not external chunks | Persistent memory **inside** one dLLM trajectory; not agent text canvas across chunks. Code: [Les1a/MetaState](https://github.com/Les1a/MetaState). |
| **MemDLM** | [2603.22241](https://arxiv.org/abs/2603.22241) | **Parametric** fast weights | Training-time inner loop; optional inference adaptation | Trajectory simulation, not chunk evidence | Closest GitHub name collision; **training method**, not agent text memory editing. [JarvisPei/MemDLM](https://github.com/JarvisPei/MemDLM). |
| **DiffResearch** | [2610.28909](https://arxiv.org/abs/2610.28909) | Multi-agent scaffold; no fixed memory canvas | dLLM for plan/write stages | Retrieval + prompts | dLLM for deep research **latency**; no memory canvas editing. [alanrbtx/DiffResearch](https://github.com/alanrbtx/DiffResearch) (Apache-2.0). |
| **RemeDi** | [2509.23653](https://arxiv.org/abs/2509.23653) | Single generation sequence | Yes — confidence remask within denoising | **Self**-confidence at same step | palimdiff **confidence baseline** is analogous; RemeDi does not condition remask on **new external chunk** vs previous. [maple-research-lab/RemeDi](https://github.com/maple-research-lab/RemeDi). |

## Dropped / adjacent (per brief)

- **Answering While You Type** ([2610.26296](https://arxiv.org/abs/2610.26296)): streaming-input dLLM decoding—not memory editing.
- **Continuous Interaction Diffusion** ([2608.10438](https://arxiv.org/abs/2608.10438)): tool interaction architecture without the empirical memory-editing setting.

## GitHub search

Query: `diffusion language model memory agent overwrite` (2026-10-29). No repository combining **dLLM partial remask** with **long-horizon agent text memory** and **chunk-conditioned** localization. Nearest code repos: MemDLM (training), MetaState (in-trajectory latent memory), LycheeMemory (compress + gate + AR reasoner), RemeDi (within-gen remask).

## Conclusion for palimdiff

| Question | Answer |
|----------|--------|
| Does prior work do **exactly** evidence-delta remask on a fixed text memory canvas per chunk? | **No** (as of search date). |
| What remains new? | Rule-based **evidence delta** (PLL with K mask partitions, old vs new chunk conditioning) + **partial re-denoise** + matched-ρ baselines + retention probe; optional **GRPO remask head** (v2). |
| Risk of overlap | **RemeDi** (confidence remask), **Plan-and-Patch** (partial dLLM edit), **MemAgent** (same task, full AR overwrite)—palimdiff must compare against these patterns explicitly in experiments. |

## Model and dataset licenses (for experiments)

| Asset | License (verify at source before release) |
|-------|---------------------------------------------|
| LLaDA-8B-Instruct | Check Hugging Face model card (GSAI-ML/LLaDA family). |
| Dream-v0-Instruct-7B | Apache-2.0 (Dream / DreamReasoner line; confirm card for exact checkpoint). |
| Qwen2.5-7B-Instruct | Qwen license on HF model card. |
| HotpotQA | Check dataset card (used in RULER-HQA-style construction). |
| This repo | MIT (`LICENSE`). |
