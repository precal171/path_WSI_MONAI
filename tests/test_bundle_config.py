"""The bundle configs must parse, resolve, and actually train.

Valid JSON proves very little. What matters is that every ``@reference`` and
``$expression`` resolves into a real object and that the assembled trainer runs
a step, so these tests instantiate the whole graph and then run two epochs on
CPU over synthetic patches.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from monai.bundle import ConfigParser

CONFIG_NAMES = ["train.json", "inference.json", "metadata.json"]


@pytest.fixture
def train_parser(bundle_root: str, patch_manifest: str, tmp_path) -> ConfigParser:
    parser = ConfigParser()
    parser.read_config(os.path.join(bundle_root, "configs", "train.json"))
    parser.update({
        "bundle_root": bundle_root,
        "manifest": patch_manifest,
        "ckpt_dir": str(tmp_path / "models"),
        "tb_dir": str(tmp_path / "tb"),
        "device": torch.device("cpu"),
        "max_epochs": 2,
        "batch_size": 2,
        "num_workers": 0,
        "amp": False,
        "patch_size": [128, 128],
    })
    return parser


class TestConfigFiles:
    @pytest.mark.parametrize("name", CONFIG_NAMES)
    def test_is_valid_json(self, bundle_root: str, name: str):
        with open(os.path.join(bundle_root, "configs", name), encoding="utf-8") as handle:
            assert isinstance(json.load(handle), dict)

    def test_metadata_declares_io_shape(self, bundle_root: str):
        with open(os.path.join(bundle_root, "configs", "metadata.json"), encoding="utf-8") as handle:
            metadata = json.load(handle)
        assert metadata["network_data_format"]["inputs"]["image"]["num_channels"] == 3
        assert metadata["network_data_format"]["outputs"]["pred"]["num_channels"] == 1

    def test_no_stray_underscore_kwargs_in_components(self, bundle_root: str):
        """``_comment`` inside a ``_target_`` dict is passed as a kwarg and blows up.

        Only MONAI's own ``_``-prefixed keys are stripped, so this is an easy and
        very confusing mistake to make when annotating a config.
        """
        allowed = {"_target_", "_desc_", "_requires_", "_disabled_", "_mode_"}

        def walk(node):
            if isinstance(node, dict):
                if "_target_" in node:
                    bad = [k for k in node if k.startswith("_") and k not in allowed]
                    assert not bad, f"{bad} would be passed to {node['_target_']} as kwargs"
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        for name in ("train.json", "inference.json"):
            with open(os.path.join(bundle_root, "configs", name), encoding="utf-8") as handle:
                walk(json.load(handle))


class TestTrainConfig:
    @pytest.mark.parametrize(
        "key",
        ["network", "loss", "optimizer", "train#preprocessing", "validate#preprocessing",
         "train#dataloader", "train#trainer", "validate#evaluator"],
    )
    def test_component_resolves(self, train_parser: ConfigParser, key: str):
        assert train_parser.get_parsed_content(key) is not None

    def test_network_shape_matches_metadata(self, train_parser: ConfigParser):
        network = train_parser.get_parsed_content("network")
        with torch.no_grad():
            output = network(torch.randn(1, 3, 128, 128))
        assert output.shape == (1, 1, 128, 128)

    def test_preprocessing_scales_to_unit_range(self, train_parser: ConfigParser, patch_manifest: str):
        with open(patch_manifest, encoding="utf-8") as handle:
            item = json.load(handle)["training"][0]

        data = train_parser.get_parsed_content("validate#preprocessing")(dict(item))
        image, label = data["image"], data["label"]

        assert image.shape[0] == 3, "image should be channel-first RGB"
        assert label.shape[0] == 1
        assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
        assert set(torch.unique(torch.as_tensor(label)).tolist()) <= {0.0, 1.0}

    def test_two_epochs_run_and_save_weights(self, train_parser: ConfigParser, tmp_path):
        for _ in train_parser.get_parsed_content("run"):
            pass

        models = tmp_path / "models"
        saved = list(models.glob("*.pt"))
        assert saved, f"no checkpoint written to {models}"

    def test_finetune_flag_requires_a_checkpoint(self, train_parser: ConfigParser, tmp_path):
        """Fine-tuning from missing weights must fail loudly, not start from noise.

        ConfigParser wraps errors raised inside a ``$`` expression in a
        RuntimeError, so the FileNotFoundError shows up as the cause rather than
        the raised type.
        """
        train_parser.update({"finetune": True, "pretrained_ckpt": str(tmp_path / "nope.pt")})

        with pytest.raises(Exception) as excinfo:
            for _ in train_parser.get_parsed_content("run"):
                pass

        chain, error = [], excinfo.value
        while error is not None and len(chain) < 10:
            chain.append(type(error))
            error = error.__cause__
        assert FileNotFoundError in chain, f"expected a FileNotFoundError in the chain, got {chain}"


class TestInferenceConfig:
    @pytest.fixture
    def infer_parser(self, bundle_root: str) -> ConfigParser:
        parser = ConfigParser()
        parser.read_config(os.path.join(bundle_root, "configs", "inference.json"))
        parser.update({"bundle_root": bundle_root, "device": torch.device("cpu")})
        return parser

    @pytest.mark.parametrize(
        "key", ["network", "preprocessing", "array_preprocessing", "inferer", "postprocessing"]
    )
    def test_component_resolves(self, infer_parser: ConfigParser, key: str):
        assert infer_parser.get_parsed_content(key) is not None

    def test_array_preprocessing_takes_arrays_not_paths(self, infer_parser: ConfigParser):
        """batch_infer.py and the infer task feed arrays straight from the WSI reader."""
        import numpy as np

        data = infer_parser.get_parsed_content("array_preprocessing")(
            {"image": np.full((3, 64, 64), 255, dtype=np.float32)}
        )
        assert float(torch.as_tensor(data["image"]).max()) <= 1.0

    def test_inference_matches_training_normalisation(self, bundle_root: str, infer_parser, train_parser):
        """A mismatch here silently degrades every prediction, so it is worth asserting."""
        import numpy as np

        patch = np.random.default_rng(0).integers(0, 255, (3, 64, 64)).astype(np.float32)
        infer_out = infer_parser.get_parsed_content("array_preprocessing")({"image": patch.copy()})["image"]

        scale = train_parser.get_parsed_content("base_transforms")[1]
        train_out = scale({"image": patch.copy()})["image"]

        np.testing.assert_allclose(
            np.asarray(torch.as_tensor(infer_out)), np.asarray(torch.as_tensor(train_out)), rtol=1e-5
        )
