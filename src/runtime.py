from __future__ import annotations

from typing import Any

import torch


def get_model_device(model: Any) -> torch.device | None:
    """Best-effort lookup for the model's current device."""
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def resolve_device(model: Any = None) -> torch.device:
    """Resolves the runtime device, preferring the model device when known."""
    model_device = get_model_device(model)
    if model_device is not None and model_device.type != "meta":
        return model_device

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def ensure_supported_4bit_runtime(device: torch.device, load_in_4bit: bool) -> None:
    """
    Fails early when the requested quantized runtime is unsupported.

    Tiny-Reason currently relies on the CUDA-backed 4-bit path used by Unsloth.
    """
    if load_in_4bit and device.type != "cuda":
        raise RuntimeError(
            "4-bit model loading/inference currently requires a CUDA device. "
            "Set 'model.load_in_4bit: false' to use the non-quantized "
            "Transformers/PEFT fallback path on CPU or MPS."
        )
