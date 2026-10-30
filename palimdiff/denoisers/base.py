from __future__ import annotations

from palimdiff.denoisers.tiny import TinyStandInDenoiser


def get_denoiser(name: str, **kwargs):
    """
    Factory for denoiser backends.

    - ``tiny``: CPU stand-in for tests (default).
    - ``llada``: LLaDA-8B-Instruct (requires GPU + transformers; optional extra).
    - ``dream``: Dream-v0-Instruct-7B (requires GPU; optional extra).
    """
    key = name.lower().strip()
    if key in ("tiny", "mock", "standin"):
        return TinyStandInDenoiser(**kwargs)
    if key == "llada":
        from palimdiff.denoisers.llada import LLaDADenoiser

        return LLaDADenoiser(**kwargs)
    if key == "dream":
        from palimdiff.denoisers.dream import DreamDenoiser

        return DreamDenoiser(**kwargs)
    raise ValueError(f"unknown denoiser: {name}")
