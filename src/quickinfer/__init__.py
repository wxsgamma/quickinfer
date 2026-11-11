"""Quickinfer's public offline inference API."""

from quickinfer.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    # CLI help and sampling validation do not initialize CUDA libraries.
    if name == "LLM":
        from quickinfer.llm import LLM
        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
