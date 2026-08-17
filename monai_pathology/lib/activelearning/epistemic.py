"""Uncertainty-based sample selection via MC dropout.

NOT WIRED UP YET -- see the caveats below before enabling it.

The idea: run the model over a candidate region several times with dropout left
active, and measure how much the predictions disagree. High disagreement means
the model is genuinely unsure, and annotating there teaches it more than
annotating somewhere it already gets right.

Three things must be true before this is useful, which is why it ships disabled:

1. **The network needs dropout.** ``get_network(..., dropout=0.0)`` is the
   default, and with no dropout every pass is identical and the variance is
   exactly zero. Set ``dropout`` to something like 0.2 in the bundle config and
   retrain before using this.
2. **The model has to be worth listening to.** On an untrained model,
   uncertainty is noise, and this performs worse than
   :class:`~monai_pathology.lib.activelearning.random.RandomStrategy`.
3. **It costs real compute.** Scoring means N forward passes over every
   candidate region of every unlabelled slide. On whole slides that is a batch
   job, not something to run inline while the user waits in QuPath.

To enable: finish :meth:`EpistemicStrategy.score_region`, register it in
``main.py``'s ``init_strategies``, and validate against random selection on a
held-out set before trusting it to direct annotation effort.
"""

from __future__ import annotations

import logging
from typing import Any

from .._compat import Strategy

logger = logging.getLogger(__name__)


class EpistemicStrategy(Strategy):
    """Select the unlabelled image the model is least certain about.

    Warning:
        Incomplete. :meth:`score_region` raises until implemented, and the
        strategy falls back to random selection so that registering it early
        cannot silently break the annotation loop.
    """

    def __init__(self, num_samples: int = 10, max_regions_per_slide: int = 32, seed: int | None = None) -> None:
        super().__init__("Most uncertain image (MC dropout)")
        self.num_samples = num_samples
        self.max_regions_per_slide = max_regions_per_slide
        self._seed = seed

    def score_region(self, image_path: str, region: Any, infer_task: Any) -> float:
        """Mean predictive variance over ``num_samples`` stochastic passes.

        Sketch of the implementation:

        1. Put the network in train mode *only* for its dropout layers, so
           batch-norm statistics are not disturbed::

               for module in network.modules():
                   if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                       module.train()

        2. Run ``num_samples`` forward passes over the same patch, collecting
           sigmoid probabilities.
        3. Return the mean per-pixel variance, or the mean entropy.
        """
        raise NotImplementedError(
            "EpistemicStrategy.score_region is not implemented yet. See this module's "
            "docstring for what has to be true before uncertainty sampling helps."
        )

    def __call__(self, request: dict[str, Any], datastore: Any) -> dict[str, Any] | None:
        logger.warning(
            "EpistemicStrategy is not implemented; falling back to random selection. "
            "See monai_pathology/lib/activelearning/epistemic.py."
        )
        from .random import RandomStrategy

        return RandomStrategy(seed=self._seed)(request, datastore)
