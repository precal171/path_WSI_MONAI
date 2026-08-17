"""Tissue-region segmentation -- SCAFFOLD, not finished.

Multi-class semantic segmentation of tissue compartments (tumour, stroma,
necrosis, ...) at low magnification, as opposed to per-nucleus segmentation.

What already works for this task without new code:

- ``scripts/network.py`` builds the network (``task_type="tissue_segmentation"``,
  ``num_classes=N``).
- ``geojson_utils`` handles multi-class masks natively -- pass a ``label_map``
  with one entry per class, and use ``separate_instances=False`` on the way out
  so each region stays a single polygon instead of fragmenting.
- ``prepare_training_data.py`` accepts multiple ``--labels``.

What still has to be written:

1. **A bundle.** Copy ``bundles/nuclei_segmentation`` to
   ``bundles/tissue_segmentation`` and change: ``num_classes`` to the class
   count, the loss to ``DiceCELoss(softmax=True, to_onehot_y=True)`` instead of
   ``sigmoid=True``, and the postprocessing to ``AsDiscreted(argmax=True)``
   rather than a threshold.
2. **Label handling in prepare_training_data.** Masks are currently written
   binarised (``mask > 0``). Multi-class needs the raw class indices preserved.
3. **An infer task**, mirroring ``infers/nuclei_segmentation.py`` but taking an
   argmax across channels rather than thresholding one.
4. **A sensible downsample.** Regions are usually annotated at 5-10x, not full
   resolution, so ``downsample`` around 4-8 is the right default here.

Until then, instantiating this raises rather than silently producing a model
that cannot train.
"""

from __future__ import annotations

from typing import Any

from .._compat import TaskConfig


class TissueSegmentation(TaskConfig):
    """Multi-class tissue-region segmentation. Not implemented yet."""

    LABELS = {"Tumor": 1, "Stroma": 2, "Necrosis": 3}

    def init(self, name: str, model_dir: str, conf: dict[str, str], planner: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "tissue_segmentation is scaffolded but not implemented. See the steps in "
            "monai_pathology/lib/configs/tissue_segmentation.py. Run with "
            "`--conf models nuclei_segmentation` in the meantime."
        )

    def infer(self) -> dict[str, Any]:
        return {}

    def trainer(self) -> Any:
        return None

    def strategy(self) -> dict[str, Any]:
        return {}
