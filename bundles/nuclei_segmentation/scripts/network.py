"""Network factory shared by every task type in this project.

The bundle configs could instantiate ``BasicUNet`` directly, but routing through
a factory is what lets the tissue-segmentation and tile-classification bundles
reuse this file: they differ mainly in the head and the number of outputs, not
in anything the rest of the pipeline cares about.

Supported task types:

``nuclei_segmentation``
    Binary 2D segmentation of nuclei. One output channel, trained with a
    sigmoid + DiceCE loss. This is the reference task -- the one that is fully
    wired up and tested.

``tissue_segmentation``
    Multi-class 2D semantic segmentation (tumour / stroma / necrosis / ...).
    Same backbone, ``num_classes`` output channels, softmax at inference.

``tile_classification``
    Whole-tile label rather than a per-pixel mask. Uses a plain classifier
    backbone instead of a UNet.

The latter two are wired far enough to build a network and run a forward pass,
but their data pipelines are not implemented -- see docs/architecture.md.
"""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from monai.networks.nets import BasicUNet, DenseNet121

#: Default UNet feature widths. Six entries: five encoder stages plus the final
#: decoder width, which is what ``BasicUNet`` expects.
DEFAULT_FEATURES: Sequence[int] = (32, 32, 64, 128, 256, 32)


def get_network(
    task_type: str = "nuclei_segmentation",
    in_channels: int = 3,
    num_classes: int = 1,
    features: Sequence[int] = DEFAULT_FEATURES,
    dropout: float = 0.0,
) -> nn.Module:
    """Build the network for ``task_type``.

    Args:
        task_type: one of ``nuclei_segmentation``, ``tissue_segmentation``,
            ``tile_classification``.
        in_channels: input channels. 3 for RGB histopathology patches.
        num_classes: output channels. 1 for binary segmentation (sigmoid),
            N for multi-class (softmax), N for classification logits.
        features: UNet feature widths; ignored for classification.
        dropout: dropout probability. Must be > 0 if you intend to use the
            MC-dropout active-learning strategy, which needs stochastic forward
            passes at inference time to estimate uncertainty.

    Returns:
        An un-initialised :class:`torch.nn.Module` on the CPU.
    """
    if task_type in ("nuclei_segmentation", "tissue_segmentation"):
        return BasicUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=num_classes,
            features=tuple(features),
            dropout=dropout,
        )

    if task_type == "tile_classification":
        return DenseNet121(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=num_classes,
            dropout_prob=dropout,
        )

    raise ValueError(
        f"Unknown task_type {task_type!r}. Expected one of: "
        "'nuclei_segmentation', 'tissue_segmentation', 'tile_classification'."
    )
