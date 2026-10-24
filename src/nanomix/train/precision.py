"""Device choice and bf16 autocast."""

import contextlib

import torch


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(precision, device):
    """Downgrade to fp32 where bf16 is not actually supported.

    Pre-Ampere CUDA cards have no bf16 path; silently running there would be
    slower than fp32, not faster.
    """
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("bf16 unsupported on this GPU (pre-Ampere); falling back to fp32")
        return "fp32"
    return precision


def autocast_ctx(device, precision):
    """Context manager wrapping forward + loss.

    bf16 carries fp32's exponent range, so unlike fp16 there are no gradient
    underflow problems and no GradScaler is needed. Backward is deliberately
    left outside: autocast covers the forward ops, and autograd then runs each
    backward in the dtype its forward was recorded in.

    No TF32 either (PyTorch's default for matmuls is off): training matmuls already run in bf16,
    so TF32 wouldn't speed them up; its only effect would be on the fp32 holdout BPB eval, which
    must stay true fp32 to be comparable with the Mac (CLAUDE.md).
    """
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
