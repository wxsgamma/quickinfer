"""Quickinfer's public offline inference API."""

from quickinfer.sampling_params import SamplingParams

__all__ = ["LLM", "PDPool", "SamplingParams"]


def __getattr__(name):
    # CLI help and sampling validation do not initialize CUDA libraries.
    if name == "LLM":
        from quickinfer.llm import LLM
        return LLM
    if name == "PDPool":
        from quickinfer.llmengine.pd_pool import PDPool
        return PDPool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
