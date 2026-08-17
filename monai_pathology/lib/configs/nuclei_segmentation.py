"""Task configuration for nuclei segmentation.

A ``TaskConfig`` is MONAI Label's unit of pluggability: it declares the labels,
locates the model weights, and builds the infer / train / strategy objects that
``main.py`` registers with the server. Adding a second task to this app means
adding a sibling module here and listing it in ``main.py`` -- nothing else in
the app needs to change.

Everything model-shaped is delegated to the bundle in ``bundles/nuclei_segmentation``
so that the interactive server and the SLURM batch jobs cannot drift apart.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .._compat import TaskConfig
from ..infers.nuclei_segmentation import NucleiSegmentationInfer
from ..trainers.nuclei_segmentation import NucleiSegmentationTrain

logger = logging.getLogger(__name__)


class NucleiSegmentation(TaskConfig):
    """Binary nucleus segmentation, backed by the nuclei_segmentation bundle."""

    #: Class name -> mask value. Matches the bundle's label space, and is what
    #: QuPath shows in its annotation class list.
    LABELS = {"Nucleus": 1}

    def init(self, name: str, model_dir: str, conf: dict[str, str], planner: Any, **kwargs: Any) -> None:
        """Called by MONAI Label at startup.

        Args:
            name: the model name the server registers this under.
            model_dir: MONAI Label's own model directory. We ignore it in favour
                of the bundle's ``models/`` directory, so that weights produced
                by a SLURM run and weights produced by an interactive fine-tune
                land in the same place.
            conf: ``--conf key value`` pairs from the ``monailabel start_server``
                command line.
            planner: MONAI Label's experiment planner. Unused here -- our patch
                geometry comes from the bundle config, not from auto-planning.
        """
        super().init(name, model_dir, conf, planner, **kwargs)

        self.name = name
        self.labels = self.LABELS
        self.conf = conf or {}

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        self.bundle_root = self.conf.get(
            "bundle_root", os.path.join(repo_root, "bundles", "nuclei_segmentation")
        )
        self.bundle_root = os.path.abspath(self.bundle_root)
        if not os.path.isdir(self.bundle_root):
            raise FileNotFoundError(
                f"Bundle not found at {self.bundle_root}. Pass a different location with "
                "`--conf bundle_root /path/to/bundles/nuclei_segmentation`."
            )

        self.model_dir = os.path.join(self.bundle_root, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.path = os.path.join(self.model_dir, "model.pt")

        # Patch geometry for interactive inference. Larger is fewer round trips
        # but more GPU memory; these defaults suit a 16 GB card comfortably.
        self.patch_size = int(self.conf.get("patch_size", 256))
        self.roi_size = (self.patch_size, self.patch_size)
        self.downsample = float(self.conf.get("downsample", 1.0))
        self.backend = self.conf.get("wsi_backend", "OpenSlide")

        if not os.path.exists(self.path):
            logger.warning(
                "No weights at %s yet. Inference will return nothing useful until you "
                "train -- annotate a few regions in QuPath and run a training round.",
                self.path,
            )

    def infer(self) -> dict[str, Any]:
        return {
            self.name: NucleiSegmentationInfer(
                path=self.path,
                bundle_root=self.bundle_root,
                labels=self.labels,
                roi_size=self.roi_size,
                downsample=self.downsample,
                backend=self.backend,
            )
        }

    def trainer(self) -> Any:
        return NucleiSegmentationTrain(
            model_dir=self.model_dir,
            bundle_root=self.bundle_root,
            labels=self.labels,
            patch_size=self.patch_size,
            downsample=self.downsample,
            backend=self.backend,
            description="Fine-tune nucleus segmentation on newly annotated regions",
        )

    def strategy(self) -> dict[str, Any]:
        # Registered centrally in main.py so that every task shares one set of
        # strategies rather than each defining its own.
        return {}
