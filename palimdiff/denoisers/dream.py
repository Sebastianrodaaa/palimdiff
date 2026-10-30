"""Dream-v0-Instruct-7B backend (optional; Apache-2.0, requires GPU)."""

from __future__ import annotations

from palimdiff.denoisers.tiny import TinyStandInDenoiser


class DreamDenoiser(TinyStandInDenoiser):
    """Placeholder — swap in Dream inference when GPU stack is configured."""

    def __init__(self, model_id: str = "Dream-org/Dream-v0-Instruct-7B", **kwargs) -> None:
        self.model_id = model_id
        print(
            f"[palimdiff] Dream backend: configure {model_id} on GPU; "
            "using TinyStandIn (MOCK) in this build."
        )
        super().__init__(**kwargs)
