"""Checkpoint helpers shared by the bundle configs, the MONAI Label trainer and
``scripts/batch_infer.py``.

The reason this exists rather than a bare ``torch.load`` in the configs: weights
reach ``models/`` by three different routes in this project -- Ignite's
``CheckpointSaver`` (which wraps the state dict in ``{"model": ...}``), a plain
``torch.save(model.state_dict())``, and TorchScript export -- and inference code
should not care which one produced the file it was handed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_weights(
    network: nn.Module,
    path: str | os.PathLike,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load weights from ``path`` into ``network``, in place.

    Accepts either a raw ``state_dict`` or an Ignite-style checkpoint dict with
    the weights under a ``"model"`` (or ``"network"``) key.

    Raises:
        FileNotFoundError: if ``path`` does not exist. This is deliberately not
            silently ignored -- a fine-tune that quietly starts from random
            weights is far worse than one that refuses to start.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train the model first "
            "(slurm/train_bundle.sbatch), or set finetune=false to train from scratch."
        )

    obj: Any = torch.load(path, map_location=device, weights_only=False)

    if isinstance(obj, dict):
        for key in ("model", "network", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise ValueError(
            f"{path} does not contain a state dict (got {type(obj).__name__}). "
            "If this is a TorchScript file, load it with torch.jit.load instead."
        )

    missing, unexpected = network.load_state_dict(obj, strict=strict)
    if missing:
        logger.warning("Checkpoint %s is missing keys: %s", path, sorted(missing))
    if unexpected:
        logger.warning("Checkpoint %s has unexpected keys: %s", path, sorted(unexpected))

    logger.info("Loaded weights from %s", path)
    return network
