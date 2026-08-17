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

    Args:
        strict: require the checkpoint to match the network exactly. On a
            mismatch this raises with the offending key names; pass ``False`` to
            load the overlapping weights and only warn.

    Raises:
        FileNotFoundError: if ``path`` does not exist. This is deliberately not
            silently ignored -- a fine-tune that quietly starts from random
            weights is far worse than one that refuses to start.
        RuntimeError: if ``strict`` and the checkpoint does not match the network.
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

    # Report *which* keys disagree rather than letting torch raise a bare
    # RuntimeError. Shape mismatches are filtered out before loading, because
    # load_state_dict raises on those even with strict=False -- and a changed
    # num_classes or in_channels is the most likely reason a checkpoint stops
    # fitting, so it is precisely the case worth naming.
    model_state = network.state_dict()
    mismatched = {
        key: (tuple(value.shape), tuple(model_state[key].shape))
        for key, value in obj.items()
        if key in model_state and hasattr(value, "shape")
        and tuple(value.shape) != tuple(model_state[key].shape)
    }

    result = network.load_state_dict(
        {k: v for k, v in obj.items() if k not in mismatched}, strict=False
    )
    missing = sorted(set(result.missing_keys) - set(mismatched))
    unexpected = sorted(result.unexpected_keys)

    if missing or unexpected or mismatched:
        detail = []
        if mismatched:
            shapes = ", ".join(
                f"{k} checkpoint{ckpt} vs network{net}"
                for k, (ckpt, net) in list(mismatched.items())[:3]
            )
            detail.append(f"{len(mismatched)} shape mismatch(es): {shapes}")
        if missing:
            detail.append(f"{len(missing)} missing key(s), e.g. {missing[:5]}")
        if unexpected:
            detail.append(f"{len(unexpected)} unexpected key(s), e.g. {unexpected[:5]}")
        summary = f"Checkpoint {path} does not match this network: " + "; ".join(detail)

        if strict:
            raise RuntimeError(
                summary + ". This usually means the checkpoint was trained with a "
                "different task_type, in_channels or num_classes than the config now "
                "asks for. Check the bundle config, or pass strict=False to load the "
                "matching weights only."
            )
        logger.warning("%s -- loading the matching weights only (strict=False).", summary)

    logger.info("Loaded weights from %s", path)
    return network
