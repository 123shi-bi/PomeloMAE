"""PomeloMAE training and evaluation."""

from collections import abc
from importlib import import_module
import sys
from types import ModuleType


def _import_legacy_timm():
    """Load timm 0.3.2, whose helper imports an alias removed in PyTorch 2."""
    try:
        import_module("torch._six")
    except ModuleNotFoundError as error:
        if error.name != "torch._six":
            raise
        legacy_torch = ModuleType("torch._six")
        legacy_torch.container_abcs = abc
        sys.modules["torch._six"] = legacy_torch
        try:
            import_module("timm")
        finally:
            del sys.modules["torch._six"]


_import_legacy_timm()
