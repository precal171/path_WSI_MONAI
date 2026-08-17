"""Interactive fine-tuning triggered from QuPath.

When the user clicks Train, this converts whatever they have annotated so far
into patches and runs a short fine-tune starting from the current weights.

It does not define a training loop. It loads the bundle's ``train.json`` and
overrides a handful of keys -- fewer epochs, ``finetune=true``, the freshly
built manifest. That is the whole point of the bundle-centred design: the model,
loss, augmentation and optimiser used for a thirty-second interactive update are
byte-for-byte the ones used by an overnight SLURM run, so improvements the user
sees in QuPath are real and not an artefact of a second, subtly different
training path.

Under a BioHPC-only deployment the server is itself running on a GPU node, so
"interactive" training here still uses a cluster GPU -- there is no local
fallback and no weights to copy anywhere.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any

from .._compat import BasicTrainTask, DefaultLabelTag

logger = logging.getLogger(__name__)


class NucleiSegmentationTrain(BasicTrainTask):
    """Short fine-tune over newly annotated regions."""

    def __init__(
        self,
        model_dir: str,
        bundle_root: str,
        labels: dict[str, int] | None = None,
        patch_size: int = 256,
        downsample: float = 1.0,
        backend: str = "OpenSlide",
        max_epochs: int = 20,
        train_batch_size: int = 8,
        learning_rate: float = 1e-4,
        val_split: float = 0.2,
        description: str = "Fine-tune nucleus segmentation",
        **kwargs: Any,
    ) -> None:
        self.bundle_root = bundle_root
        self.labels = labels or {"Nucleus": 1}
        self.patch_size = patch_size
        self.downsample = downsample
        self.backend = backend
        self.learning_rate = learning_rate
        self.val_split = val_split

        # BasicTrainTask takes model_dir, description, labels and a config dict --
        # but not max_epochs or batch size, which it expects to come through the
        # per-request `config`. We keep our own copies as the defaults that the
        # bundle overrides are built from.
        self.max_epochs = max_epochs
        self.train_batch_size = train_batch_size
        self._parser = None

        super().__init__(
            model_dir=model_dir,
            description=description,
            labels=list(self.labels.keys()),
            config={
                "max_epochs": max_epochs,
                "train_batch_size": train_batch_size,
                "learning_rate": learning_rate,
            },
            **kwargs,
        )

    # ------------------------------------------------------------------
    # BasicTrainTask abstract members
    # ------------------------------------------------------------------
    # Training here is driven by running the bundle (see run_bundle_training),
    # not by the base class assembling these pieces itself. They are still
    # implemented -- the base class declares them abstract, so the task cannot be
    # instantiated otherwise -- and each one resolves from the very same
    # train.json, so anything reading them sees exactly what training used.

    def _bundle_parser(self):
        """A ConfigParser over the bundle's train.json, built once."""
        if getattr(self, "_parser", None) is None:
            from monai.bundle import ConfigParser

            if self.bundle_root not in sys.path:
                sys.path.insert(0, self.bundle_root)

            parser = ConfigParser()
            parser.read_config(os.path.join(self.bundle_root, "configs", "train.json"))
            parser.update({"bundle_root": self.bundle_root})
            self._parser = parser
        return self._parser

    def network(self, context: Any = None) -> Any:
        return self._bundle_parser().get_parsed_content("network")

    def optimizer(self, context: Any = None) -> Any:
        return self._bundle_parser().get_parsed_content("optimizer")

    def loss_function(self, context: Any = None) -> Any:
        return self._bundle_parser().get_parsed_content("loss")

    def train_pre_transforms(self, context: Any = None) -> Any:
        return self._bundle_parser().get_parsed_content("train#preprocessing")

    def train_post_transforms(self, context: Any = None) -> Any:
        return self._bundle_parser().get_parsed_content("train#postprocessing")

    def val_inferer(self, context: Any = None) -> Any:
        return self._bundle_parser().get_parsed_content("validate#inferer")

    def train_data_loader(self, context: Any = None, num_workers: int = 0, shuffle: bool = False) -> Any:
        return self._bundle_parser().get_parsed_content("train#dataloader")

    # ------------------------------------------------------------------
    # Turning stored annotations into a manifest
    # ------------------------------------------------------------------

    def build_manifest(self, datastore: Any, output_dir: str, request: dict[str, Any] | None = None) -> str:
        """Cut patches from every labelled image in the datastore.

        Args:
            datastore: MONAI Label datastore holding images and their labels.
            output_dir: where patches and ``manifest.json`` are written.
            request: the training request; may carry overrides such as
                ``patch_size``.

        Returns:
            Path to the manifest.

        Raises:
            ValueError: if nothing has been annotated yet -- a clearer failure
                than letting training start on an empty dataset.
        """
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from tools.prepare_training_data import process_slide, _split  # noqa: PLC0415

        request = request or {}
        patch_size = int(request.get("patch_size", self.patch_size))
        labels = list(self.labels.keys())

        image_ids = datastore.get_labeled_images()
        if not image_ids:
            raise ValueError(
                "No annotations have been submitted yet. Draw and submit at least one "
                "region in QuPath before training."
            )

        entries: list[dict[str, str]] = []
        for image_id in image_ids:
            slide_path = datastore.get_image_uri(image_id)
            label_path = datastore.get_label_uri(image_id, DefaultLabelTag.FINAL.value)
            if not slide_path or not label_path or not os.path.exists(label_path):
                logger.warning("Skipping %s: no readable label file.", image_id)
                continue

            entries.extend(
                process_slide(
                    slide_path=slide_path,
                    annotation_path=label_path,
                    output_dir=output_dir,
                    labels=labels,
                    roi_labels=[],
                    patch_size=patch_size,
                    stride=patch_size,
                    downsample=self.downsample,
                    background_fraction=0.1,
                    backend=self.backend,
                )
            )

        if not entries:
            raise ValueError(
                f"No patches could be cut from {len(image_ids)} labelled image(s). "
                f"Check that your QuPath annotations use one of these classes: {labels}"
            )

        train, val = _split(entries, self.val_split, seed=0)
        manifest = {
            "task": "nuclei_segmentation",
            "created": datetime.now(timezone.utc).isoformat(),
            "labels": labels,
            "patch_size": [patch_size, patch_size],
            "downsample": self.downsample,
            "num_patches": len(entries),
            "training": train,
            "validation": val or train[:1],  # a fine-tune on one region can leave val empty
        }

        import json

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        logger.info("Built manifest with %d patches from %d image(s)", len(entries), len(image_ids))
        return manifest_path

    # ------------------------------------------------------------------
    # Running the bundle
    # ------------------------------------------------------------------

    def run_bundle_training(self, manifest_path: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the bundle's ``train.json`` against ``manifest_path``."""
        from monai.bundle import ConfigParser

        request = request or {}
        if self.bundle_root not in sys.path:
            sys.path.insert(0, self.bundle_root)

        model_dir = self.model_dir if isinstance(self.model_dir, str) else str(self.model_dir)
        finetune = os.path.exists(os.path.join(model_dir, "model.pt"))

        overrides: dict[str, Any] = {
            "bundle_root": self.bundle_root,
            "ckpt_dir": model_dir,
            "manifest": manifest_path,
            "max_epochs": int(request.get("max_epochs", self.max_epochs)),
            "batch_size": int(request.get("train_batch_size", self.train_batch_size)),
            "learning_rate": float(request.get("learning_rate", self.learning_rate)),
            "num_workers": int(request.get("num_workers", 2)),
            "finetune": bool(request.get("finetune", finetune)),
        }
        logger.info("Fine-tuning from existing weights" if overrides["finetune"] else "Training from scratch")

        parser = ConfigParser()
        parser.read_config(os.path.join(self.bundle_root, "configs", "train.json"))
        parser.update(overrides)

        for statement in parser.get_parsed_content("run"):
            _ = statement  # each entry is evaluated by ConfigParser on access

        return {"manifest": manifest_path, **overrides}

    def __call__(self, request: dict[str, Any], datastore: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Entry point invoked when the user triggers training.

        Patches are written to a temporary directory by default and cleaned up
        afterwards, since an interactive round is throwaway data. Pass
        ``keep_patches`` to retain them for inspection.
        """
        request = request or {}
        keep = bool(request.get("keep_patches", False))
        patch_dir = request.get("patch_dir") or tempfile.mkdtemp(prefix="monailabel_patches_")

        try:
            manifest_path = self.build_manifest(datastore, patch_dir, request)
            result = self.run_bundle_training(manifest_path, request)
            result["patch_dir"] = patch_dir if keep else None
            return result
        finally:
            if not keep and not request.get("patch_dir"):
                shutil.rmtree(patch_dir, ignore_errors=True)
