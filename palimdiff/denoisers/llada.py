"""LLaDA-8B-Instruct backend (optional; requires GPU + transformers)."""

from __future__ import annotations

from palimdiff.denoisers.tiny import TinyStandInDenoiser


class LLaDADenoiser(TinyStandInDenoiser):
    """
    Placeholder wrapper: loads GSAI-ML/LLaDA-8B-Instruct when dependencies exist.

    Until ``transformers`` and weights are available, falls back to TinyStandIn with
    a warning printed once at init.
    """

    def __init__(self, model_id: str = "GSAI-ML/LLaDA-8B-Instruct", **kwargs) -> None:
        self.model_id = model_id
        self._using_fallback = True
        try:
            import torch  # noqa: F401
            from transformers import AutoModel  # noqa: F401
        except ImportError:
            print(
                "[palimdiff] LLaDA backend: missing torch/transformers; using TinyStandIn (MOCK)."
            )
        else:
            print(
                f"[palimdiff] LLaDA backend: install weights for {model_id} on GPU; "
                "currently TinyStandIn (MOCK) until load is implemented."
            )
        super().__init__(**kwargs)
