"""Resolve MONAI Label's base classes across releases.

MONAI Label has moved these types between ``monailabel.interfaces.*`` and
``monailabel.tasks.*`` more than once, and the exact layout depends on which
version happens to be in your container. Importing them directly from a guessed
path means that a version bump breaks every module in ``lib/`` at once, with a
traceback that points at whichever file Python happened to load first rather
than at the actual problem.

So every app module imports its base classes from here instead. When a version
bump moves something, this is the only file to fix, and the error message says
so directly.

If you hit :class:`MonaiLabelImportError`, run::

    python -c "import monailabel, pkgutil; print(monailabel.__version__); \\
        print([m.name for m in pkgutil.walk_packages(monailabel.__path__, 'monailabel.')])"

and add the path you find to the relevant candidate list below.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "MonaiLabelImportError",
    "MONAILABEL_AVAILABLE",
    "MONAILabelApp",
    "TaskConfig",
    "BasicInferTask",
    "InferType",
    "BasicTrainTask",
    "Strategy",
    "Datastore",
    "DefaultLabelTag",
    "strtobool",
]


class MonaiLabelImportError(ImportError):
    """MONAI Label is installed but a class was not where we expected it."""


def _first_available(candidates: Sequence[tuple[str, str]], what: str) -> Any:
    """Return the first importable ``(module, attribute)`` pair.

    Args:
        candidates: ``(module_path, attribute_name)`` pairs, most-likely first.
        what: human-readable name, used in the error message.
    """
    tried: list[str] = []
    for module_path, attribute in candidates:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            tried.append(f"{module_path} (no such module)")
            continue
        obj = getattr(module, attribute, None)
        if obj is not None:
            logger.debug("Resolved %s to %s.%s", what, module_path, attribute)
            return obj
        tried.append(f"{module_path} (no attribute {attribute!r})")

    raise MonaiLabelImportError(
        f"Could not locate {what} in the installed monailabel.\nTried:\n  "
        + "\n  ".join(tried)
        + "\n\nYour MONAI Label version probably moved it. See the guidance in "
        "monai_pathology/lib/_compat.py for how to find the new path."
    )


try:
    import monailabel  # noqa: F401

    MONAILABEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in slim environments
    MONAILABEL_AVAILABLE = False


if MONAILABEL_AVAILABLE:
    MONAILabelApp = _first_available(
        [
            ("monailabel.interfaces.app", "MONAILabelApp"),
            ("monailabel.interfaces", "MONAILabelApp"),
        ],
        "MONAILabelApp",
    )

    TaskConfig = _first_available(
        [
            ("monailabel.interfaces.config", "TaskConfig"),
            ("monailabel.interfaces.tasks.config", "TaskConfig"),
            ("monailabel.tasks.config", "TaskConfig"),
        ],
        "TaskConfig",
    )

    BasicInferTask = _first_available(
        [
            ("monailabel.tasks.infer.basic_infer", "BasicInferTask"),
            ("monailabel.interfaces.tasks.infer_v2", "BasicInferTask"),
            ("monailabel.interfaces.tasks.infer", "InferTask"),
        ],
        "BasicInferTask",
    )

    InferType = _first_available(
        [
            ("monailabel.interfaces.tasks.infer_v2", "InferType"),
            ("monailabel.tasks.infer.basic_infer", "InferType"),
            ("monailabel.interfaces.tasks.infer", "InferType"),
        ],
        "InferType",
    )

    BasicTrainTask = _first_available(
        [
            ("monailabel.tasks.train.basic_train", "BasicTrainTask"),
            ("monailabel.interfaces.tasks.train", "BasicTrainTask"),
        ],
        "BasicTrainTask",
    )

    Strategy = _first_available(
        [
            ("monailabel.interfaces.tasks.strategy", "Strategy"),
            ("monailabel.tasks.activelearning.strategy", "Strategy"),
        ],
        "Strategy",
    )

    Datastore = _first_available(
        [
            ("monailabel.interfaces.datastore", "Datastore"),
            ("monailabel.datastore", "Datastore"),
        ],
        "Datastore",
    )

    DefaultLabelTag = _first_available(
        [
            ("monailabel.interfaces.datastore", "DefaultLabelTag"),
            ("monailabel.datastore", "DefaultLabelTag"),
        ],
        "DefaultLabelTag",
    )

    try:
        from monailabel.utils.others.generic import strtobool
    except ImportError:  # pragma: no cover

        def strtobool(value: Any) -> bool:  # type: ignore[misc]
            """Fallback for older/newer layouts that do not export ``strtobool``."""
            return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

else:
    # Import-time placeholders so that `import monai_pathology.main` fails with a
    # clear message at *use* time rather than an opaque ImportError at import
    # time. This is what lets the test suite import and inspect these modules in
    # an environment without monailabel (see tests/test_app_imports.py).
    _MESSAGE = (
        "monailabel is not installed. It is only needed to run the annotation "
        "server; training and batch inference do not require it. "
        "Install it with: pip install -r requirements-train.txt"
    )

    class _Missing:
        """Raises a useful error the moment anything tries to use it."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise MonaiLabelImportError(_MESSAGE)

        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)

    MONAILabelApp = type("MONAILabelApp", (_Missing,), {})
    TaskConfig = type("TaskConfig", (_Missing,), {})
    BasicInferTask = type("BasicInferTask", (_Missing,), {})
    BasicTrainTask = type("BasicTrainTask", (_Missing,), {})
    Strategy = type("Strategy", (_Missing,), {})
    Datastore = type("Datastore", (_Missing,), {})

    class DefaultLabelTag:  # type: ignore[no-redef]
        """Mirror of MONAI Label's label tags, for slim environments."""

        ORIGINAL = "original"
        FINAL = "final"

    class InferType:  # type: ignore[no-redef]
        """Mirror of MONAI Label's InferType constants, for slim environments."""

        SEGMENTATION = "segmentation"
        ANNOTATION = "annotation"
        CLASSIFICATION = "classification"
        DEEPGROW = "deepgrow"
        DEEPEDIT = "deepedit"
        OTHERS = "others"

    def strtobool(value: Any) -> bool:  # type: ignore[misc]
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")
