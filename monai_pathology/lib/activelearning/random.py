"""Random next-sample selection -- the baseline active-learning strategy.

Deliberately the default. Random selection is the control that any smarter
strategy has to beat, and on a fresh project it is genuinely the right choice:
until the model has learned something, its uncertainty estimates are noise, and
"uncertainty sampling" on an untrained model just picks whatever it is
arbitrarily confused by.

Switch to :mod:`epistemic` once you have a model that produces sensible
predictions, and compare against this.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from .._compat import Strategy

logger = logging.getLogger(__name__)


class RandomStrategy(Strategy):
    """Pick an unlabelled image uniformly at random."""

    def __init__(self, seed: int | None = None) -> None:
        super().__init__("Random unlabelled image")
        self._rng = random.Random(seed)

    def __call__(self, request: dict[str, Any], datastore: Any) -> dict[str, Any] | None:
        images = datastore.get_unlabeled_images()
        if not images:
            logger.info("Every image already has annotations; nothing left to select.")
            return None

        images.sort()  # sort first so a fixed seed gives a reproducible sequence
        image_id = self._rng.choice(images)
        logger.info("Next sample: %s (%d unlabelled remaining)", image_id, len(images))
        return {"id": image_id, "strategy": self.__class__.__name__}
