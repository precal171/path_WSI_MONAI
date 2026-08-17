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
``--conf models nuclei_segmentation`` (or ``--conf models all``, which loads
every *finished* task -- see :data:`SCAFFOLDED_TASKS`).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

# MONAI Label's app loader execs this file as a *top-level* module rather than
# importing it as part of the monai_pathology package, so relative imports
# ("from .lib import ...") raise "attempted relative import with no known parent
# package" and the server never starts. Putting the package's parent directory
# on sys.path lets the absolute imports below work under both entry points:
# `monailabel start_server --app monai_pathology` and `import monai_pathology.main`.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_PARENT = os.path.dirname(_PACKAGE_DIR)
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from monai_pathology.lib._compat import MONAILabelApp  # noqa: E402
from monai_pathology.lib.activelearning.random import RandomStrategy  # noqa: E402
from monai_pathology.lib.configs.nuclei_segmentation import NucleiSegmentation  # noqa: E402

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
    from monai_pathology.lib.configs.tissue_segmentation import TissueSegmentation

    AVAILABLE_TASKS["tissue_segmentation"] = TissueSegmentation
except Exception:  # noqa: BLE001
    logger.debug("tissue_segmentation task not available", exc_info=True)

try:
    from monai_pathology.lib.configs.tile_classification import TileClassification

    AVAILABLE_TASKS["tile_classification"] = TileClassification
except Exception:  # noqa: BLE001
    logger.debug("tile_classification task not available", exc_info=True)

#: Tasks that import and register but whose data pipelines are unfinished --
#: their TaskConfig.init() raises NotImplementedError. They are excluded from
#: `--conf models all` so that it cannot kill the server at construction; naming
#: one explicitly still works and still raises, which is the useful behaviour
#: for whoever is actually building that task out.
SCAFFOLDED_TASKS = frozenset({"tissue_segmentation", "tile_classification"})

DEFAULT_TASKS = ["nuclei_segmentation"]


class MyApp(MONAILabelApp):
    """Pathology WSI app: interactive segmentation plus active learning."""

    def __init__(self, app_dir: str, studies: str, conf: dict[str, str] | None = None, **kwargs: Any) -> None:
        self.conf = conf or {}
        self.app_dir = app_dir
        self.studies = studies

        requested = self.conf.get("models", ",".join(DEFAULT_TASKS))
        if requested.strip() == "all":
            names = [n for n in AVAILABLE_TASKS if n not in SCAFFOLDED_TASKS]
            skipped = sorted(set(AVAILABLE_TASKS) & SCAFFOLDED_TASKS)
            if skipped:
                logger.info(
                    "`--conf models all` skipping unfinished task(s): %s. "
                    "Name one explicitly to work on it.",
                    ", ".join(skipped),
                )
        else:
            names = [n.strip() for n in requested.split(",") if n.strip()]

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
            try:
                task.init(name, model_dir, self.conf, None)
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"Task {name!r} is scaffolded but not finished, so the server cannot "
                    f"start with it: {exc}. Working task(s): "
                    f"{sorted(set(AVAILABLE_TASKS) - SCAFFOLDED_TASKS)}."
                ) from exc
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
            from monai_pathology.lib.activelearning.epistemic import EpistemicStrategy

            strategies["epistemic"] = EpistemicStrategy()
            logger.warning("Epistemic strategy enabled, but it is not implemented -- it falls back to random.")

        for name, task in self.tasks.items():
            strategies.update(task.strategy() or {})

        logger.info("Active-learning strategies: %s", sorted(strategies))
        return strategies
