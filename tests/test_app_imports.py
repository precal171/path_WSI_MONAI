"""The MONAI Label app layer: imports, wiring, and the request adapter.

These tests run with or without ``monailabel`` installed. Anything that needs
the real base classes is marked ``requires_monailabel`` and skips cleanly --
which matters because training and batch inference do not need the package at
all, and that environment should still be able to run the suite.
"""

from __future__ import annotations

import importlib
import os

import pytest

from monai_pathology.lib import _compat

requires_monailabel = pytest.mark.skipif(
    not _compat.MONAILABEL_AVAILABLE, reason="monailabel is not installed"
)

APP_MODULES = [
    "monai_pathology.main",
    "monai_pathology.lib._compat",
    "monai_pathology.lib.configs.nuclei_segmentation",
    "monai_pathology.lib.configs.tissue_segmentation",
    "monai_pathology.lib.configs.tile_classification",
    "monai_pathology.lib.infers.nuclei_segmentation",
    "monai_pathology.lib.trainers.nuclei_segmentation",
    "monai_pathology.lib.activelearning.random",
    "monai_pathology.lib.activelearning.epistemic",
    "monai_pathology.lib.utils.geojson_utils",
    "monai_pathology.lib.utils.wsi_utils",
]


class TestImports:
    @pytest.mark.parametrize("module", APP_MODULES)
    def test_module_imports(self, module: str):
        """Every module must import even with monailabel absent.

        That is the whole point of _compat: a missing or moved base class should
        surface as one clear error at construction, not as an import explosion
        across the package.
        """
        assert importlib.import_module(module) is not None

    def test_scripts_import(self):
        for module in ("tools.build_synthetic_wsi", "tools.prepare_training_data", "tools.batch_infer"):
            assert importlib.import_module(module) is not None


class TestCompatShim:
    @pytest.mark.parametrize(
        "name",
        ["MONAILabelApp", "TaskConfig", "BasicInferTask", "InferType",
         "BasicTrainTask", "Strategy", "Datastore", "DefaultLabelTag"],
    )
    def test_exports_every_base_class(self, name: str):
        assert getattr(_compat, name, None) is not None

    @requires_monailabel
    @pytest.mark.parametrize(
        "name,expected_module",
        [
            ("MONAILabelApp", "monailabel.interfaces.app"),
            ("TaskConfig", "monailabel.interfaces.config"),
            ("BasicInferTask", "monailabel.tasks.infer.basic_infer"),
            ("InferType", "monailabel.interfaces.tasks.infer_v2"),
            ("BasicTrainTask", "monailabel.tasks.train.basic_train"),
            ("Strategy", "monailabel.interfaces.tasks.strategy"),
        ],
    )
    def test_resolves_to_the_real_class(self, name: str, expected_module: str):
        """Pins the paths verified against monailabel 0.8.x.

        If this fails after a version bump, the shim needs the new path adding --
        which is exactly the signal it exists to give.
        """
        assert getattr(_compat, name).__module__ == expected_module

    def test_strtobool(self):
        assert _compat.strtobool("true") and _compat.strtobool("1") and _compat.strtobool("YES")
        assert not _compat.strtobool("false") and not _compat.strtobool("0")


class TestTaskRegistry:
    def test_nuclei_task_is_registered(self):
        from monai_pathology.main import AVAILABLE_TASKS, DEFAULT_TASKS

        assert "nuclei_segmentation" in AVAILABLE_TASKS
        assert DEFAULT_TASKS == ["nuclei_segmentation"]

    def test_scaffolded_tasks_refuse_to_initialise(self):
        """A half-built task must fail clearly rather than pretend to work."""
        from monai_pathology.lib.configs.tile_classification import TileClassification
        from monai_pathology.lib.configs.tissue_segmentation import TissueSegmentation

        for cls in (TissueSegmentation, TileClassification):
            with pytest.raises(NotImplementedError, match="scaffolded but not implemented"):
                cls().init("x", "/tmp", {}, None)


class TestInferRequestAdapter:
    """The QuPath request -> Region conversion, which needs no monailabel."""

    def test_maps_location_and_size(self):
        from monai_pathology.lib.infers.nuclei_segmentation import NucleiSegmentationInfer

        region = NucleiSegmentationInfer._region_from_request(
            {"location": [100, 200], "size": [512, 256], "level": 0}, downsample=1.0
        )
        assert (region.x, region.y, region.width, region.height) == (100, 200, 512, 256)

    def test_pyramid_level_becomes_downsample(self):
        from monai_pathology.lib.infers.nuclei_segmentation import NucleiSegmentationInfer

        region = NucleiSegmentationInfer._region_from_request(
            {"location": [0, 0], "size": [512, 512], "level": 2}, downsample=1.0
        )
        assert region.downsample == 4.0

    def test_defaults_when_fields_are_absent(self):
        from monai_pathology.lib.infers.nuclei_segmentation import NucleiSegmentationInfer

        region = NucleiSegmentationInfer._region_from_request({}, downsample=1.0)
        assert region.width > 0 and region.height > 0


@requires_monailabel
class TestLiveConstruction:
    """Constructing the real subclasses against the real base classes."""

    def test_infer_task(self, bundle_root: str):
        from monai_pathology.lib.infers.nuclei_segmentation import NucleiSegmentationInfer

        task = NucleiSegmentationInfer(
            path=os.path.join(bundle_root, "models", "model.pt"), bundle_root=bundle_root
        )
        assert task.labels == {"Nucleus": 1}
        assert isinstance(task.path, list)  # BasicInferTask wraps a str path in a list

    def test_train_task_and_its_abstract_members(self, bundle_root: str):
        from monai_pathology.lib.trainers.nuclei_segmentation import NucleiSegmentationTrain

        task = NucleiSegmentationTrain(
            model_dir=os.path.join(bundle_root, "models"), bundle_root=bundle_root
        )
        # These come from the bundle, so they prove the trainer and the batch
        # job really are configured from one file.
        assert type(task.network()).__name__ == "BasicUNet"
        assert type(task.optimizer()).__name__ == "Adam"
        assert type(task.loss_function()).__name__ == "DiceCELoss"

    def test_task_config_builds_infer_and_trainer(self, bundle_root: str, tmp_path):
        from monai_pathology.lib.configs.nuclei_segmentation import NucleiSegmentation

        config = NucleiSegmentation()
        config.init("nuclei_segmentation", str(tmp_path), {"bundle_root": bundle_root}, None)

        assert config.labels == {"Nucleus": 1}
        assert "nuclei_segmentation" in config.infer()
        assert config.trainer() is not None

    def test_task_config_rejects_a_bad_bundle_path(self, tmp_path):
        from monai_pathology.lib.configs.nuclei_segmentation import NucleiSegmentation

        with pytest.raises(FileNotFoundError, match="Bundle not found"):
            NucleiSegmentation().init(
                "nuclei_segmentation", str(tmp_path), {"bundle_root": "/nonexistent"}, None
            )

    def test_random_strategy_selects_and_is_reproducible(self):
        from monai_pathology.lib.activelearning.random import RandomStrategy

        class FakeDatastore:
            def get_unlabeled_images(self):
                return ["c", "a", "b"]

        assert RandomStrategy(seed=1)({}, FakeDatastore())["id"] == \
               RandomStrategy(seed=1)({}, FakeDatastore())["id"]

    def test_random_strategy_returns_none_when_everything_is_labelled(self):
        from monai_pathology.lib.activelearning.random import RandomStrategy

        class Empty:
            def get_unlabeled_images(self):
                return []

        assert RandomStrategy()({}, Empty()) is None
