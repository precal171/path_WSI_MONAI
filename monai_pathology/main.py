"""MONAI Label application entry point.

This is what ``monailabel start_server --app monai_pathology`` loads. It
registers the inference models, training tasks and active-learning strategies
that QuPath then drives over REST.

Start it on a BioHPC GPU node with ``slurm/start_label_server.sbatch``; see
``docs/biohpc_setup.md`` for how QuPath connects to it.

Adding a task type is a two-step change and touches nothing else:

1. Write ``lib/configs/<task>.py`` with a ``TaskConfig`` subclass.
2. Add it to :data:`AVAILABLE_TASKS` below.

Which of them actually load is then chosen at runtime with
``--conf models nuclei_segmentation`` (or ``--conf models all``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .lib._compat import MONAILabelApp
from .lib.activelearning.random import RandomStrategy
from .lib.configs.nuclei_segmentation import NucleiSegmentation

logger = logging.getLogger(__name__)

#: Task name -> TaskConfig class. The single place tasks are registered.
#:
#: tissue_segmentation and tile_classification are scaffolded but not finished --
#: they build a network and parse their config, but their data pipelines are not
#: implemented. See docs/architecture.md.
AVAILABLE_TASKS: dict[str, type] = {
    "nuclei_segmentation": NucleiSegmentation,
}

try:  # optional so a half-finished task cannot stop the server from booting
    from .lib.configs.tissue_segmentation import TissueSegmentation

    AVAILABLE_TASKS["tissue_segmentation"] = TissueSegmentation
except Exception:  # noqa: BLE001
    logger.debug("tissue_segmentation task not available", exc_info=True)

try:
    from .lib.configs.tile_classification import TileClassification

    AVAILABLE_TASKS["tile_classification"] = TileClassification
except Exception:  # noqa: BLE001
    logger.debug("tile_classification task not available", exc_info=True)

DEFAULT_TASKS = ["nuclei_segmentation"]


class MyApp(MONAILabelApp):
    """Pathology WSI app: interactive segmentation plus active learning."""

    def __init__(self, app_dir: str, studies: str, conf: dict[str, str] | None = None, **kwargs: Any) -> None:
        self.conf = conf or {}
        self.app_dir = app_dir
        self.studies = studies

        requested = self.conf.get("models", ",".join(DEFAULT_TASKS))
        names = list(AVAILABLE_TASKS) if requested.strip() == "all" else [
            n.strip() for n in requested.split(",") if n.strip()
        ]

        unknown = [n for n in names if n not in AVAILABLE_TASKS]
        if unknown:
            raise ValueError(
                f"Unknown model(s) {unknown}. Available: {sorted(AVAILABLE_TASKS)}. "
                "Pass them with `--conf models <name>[,<name>]` or `--conf models all`."
            )

        model_dir = os.path.join(app_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        self.tasks: dict[str, Any] = {}
        for name in names:
            task = AVAILABLE_TASKS[name]()
            task.init(name, model_dir, self.conf, None)
            self.tasks[name] = task
            logger.info("Registered task %r", name)

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=self.conf,
            name="MONAI Pathology WSI",
            description=(
                "Whole-slide pathology annotation and active learning. Segment nuclei "
                "interactively from QuPath, fine-tune on your own annotations, and apply "
                "the result across a slide archive."
            ),
            **kwargs,
        )

    def init_infers(self) -> dict[str, Any]:
        infers: dict[str, Any] = {}
        for name, task in self.tasks.items():
            for key, value in (task.infer() or {}).items():
                infers[key] = value
                logger.info("Inference model available: %s", key)
        return infers

    def init_trainers(self) -> dict[str, Any]:
        trainers: dict[str, Any] = {}
        for name, task in self.tasks.items():
            trainer = task.trainer()
            if trainer is not None:
                trainers[name] = trainer
                logger.info("Training task available: %s", name)
        return trainers

    def init_strategies(self) -> dict[str, Any]:
        """Strategies decide which image QuPath's "Next Sample" button offers.

        Registered app-wide rather than per task, since which slide to annotate
        next is a property of the annotation campaign, not of one model.
        """
        strategies: dict[str, Any] = {"random": RandomStrategy()}

        # Epistemic (MC-dropout) selection is scaffolded but not finished; it
        # needs a dropout-enabled trained model to mean anything. Enable it
        # explicitly once that is true.
        if str(self.conf.get("epistemic", "false")).lower() in ("1", "true", "yes"):
            from .lib.activelearning.epistemic import EpistemicStrategy

            strategies["epistemic"] = EpistemicStrategy()
            logger.warning("Epistemic strategy enabled, but it is not implemented -- it falls back to random.")

        for name, task in self.tasks.items():
            strategies.update(task.strategy() or {})

        logger.info("Active-learning strategies: %s", sorted(strategies))
        return strategies
