"""Tile classification -- SCAFFOLD, not finished.

One label per tile (tumour vs benign, gradeable vs not) rather than a per-pixel
mask. Cheaper to annotate than segmentation, and often enough when the question
is "where should a pathologist look" rather than "what is the exact boundary".

This one diverges from the other two tasks more than it first appears, which is
why it is scaffolded separately rather than folded into the segmentation path:

- **Labels are points or boxes, not outlines.** In QuPath these are typically
  annotations whose class is the tile label. There is no mask to rasterise, so
  ``annotations_to_mask`` is not part of the data path at all -- the label is a
  scalar per patch.
- **The manifest carries an integer, not a mask path.** ``train.json``'s
  ``{"image": ..., "label": ...}`` entries become ``{"image": ..., "label": 2}``,
  which also means dropping ``LoadImaged`` for the label key.
- **Metrics differ.** Dice is meaningless here; use accuracy, or AUROC for the
  binary case (``monai.handlers.ROCAUC``).
- **Class imbalance is usually severe** -- tumour tiles are a small minority of
  a slide. Expect to need weighted sampling or a weighted loss.

``scripts/network.py`` already returns a DenseNet121 classifier for
``task_type="tile_classification"``, so the model side is ready; the data and
config sides are not.
"""

from __future__ import annotations

from typing import Any

from .._compat import TaskConfig


class TileClassification(TaskConfig):
    """Per-tile classification. Not implemented yet."""

    LABELS = {"Benign": 0, "Tumor": 1}

    def init(self, name: str, model_dir: str, conf: dict[str, str], planner: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "tile_classification is scaffolded but not implemented. See the notes in "
            "monai_pathology/lib/configs/tile_classification.py. Run with "
            "`--conf models nuclei_segmentation` in the meantime."
        )

    def infer(self) -> dict[str, Any]:
        return {}

    def trainer(self) -> Any:
        return None

    def strategy(self) -> dict[str, Any]:
        return {}
