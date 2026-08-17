"""The whole loop on synthetic data, with no GPU, no real slide and no server.

Slide + annotations -> patches -> training -> inference -> GeoJSON, then the
predictions are scored against the ground truth the fixture generated.

This is the test that catches integration mistakes the unit tests cannot: a
coordinate convention that is self-consistent but wrong, a normalisation that
differs between training and inference, a manifest the bundle cannot read. Each
of those leaves every unit test green and produces a model that simply does not
work.

It trains a real model for a few epochs on CPU, so it takes a minute or so.
Deselect with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def dense_slide(tmp_path_factory) -> tuple[str, str]:
    """A synthetic slide with plenty of nuclei.

    Deliberately denser than the shared ``synthetic_slide`` fixture. At the
    sparse default only ~4% of a patch is foreground, and a few CPU epochs of
    DiceCE on that imbalance collapse to predicting all background -- which
    would make this test fail for reasons that have nothing to do with whether
    the pipeline is wired up correctly. Real H&E is nucleus-dense anyway.
    """
    from tools.build_synthetic_wsi import build_slide, write_geojson, write_pyramid

    directory = tmp_path_factory.mktemp("e2e_wsi")
    slide_path = str(directory / "dense.tiff")
    geojson_path = str(directory / "dense.geojson")

    image, polygons = build_slide(width=1024, height=1024, num_nuclei=700, seed=7)
    write_pyramid(image, slide_path, tile_size=256, num_levels=2)
    write_geojson(polygons, geojson_path)
    return slide_path, geojson_path


@pytest.fixture(scope="module")
def prepared_dataset(dense_slide, tmp_path_factory) -> tuple[str, str, str]:
    """Run the real data-prep CLI over the synthetic slide."""
    from tools.prepare_training_data import main as prepare_main

    slide_path, geojson_path = dense_slide
    output_dir = str(tmp_path_factory.mktemp("e2e_patches"))

    exit_code = prepare_main([
        "--slide", slide_path,
        "--annotations", geojson_path,
        "--output-dir", output_dir,
        "--labels", "Nucleus",
        "--backend", "TiffFile",
        "--patch-size", "128",
        "--background-fraction", "0.2",
    ])
    assert exit_code == 0

    manifest_path = os.path.join(output_dir, "manifest.json")
    assert os.path.exists(manifest_path)
    return slide_path, geojson_path, manifest_path


class TestDataPreparation:
    def test_manifest_is_well_formed(self, prepared_dataset):
        _, _, manifest_path = prepared_dataset
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)

        assert manifest["training"] and manifest["validation"]
        assert manifest["labels"] == ["Nucleus"]

        for entry in manifest["training"]:
            assert os.path.isabs(entry["image"]) and os.path.exists(entry["image"])
            assert os.path.exists(entry["label"])

    def test_patches_and_masks_align(self, prepared_dataset):
        """Masks must mark the dark nuclei, not empty background.

        A coordinate bug typically shows up right here: the mask is a perfectly
        reasonable shape, just describing a different part of the slide.
        """
        from PIL import Image

        _, _, manifest_path = prepared_dataset
        with open(manifest_path, encoding="utf-8") as handle:
            entries = json.load(handle)["training"]

        checked = 0
        for entry in entries:
            image = np.asarray(Image.open(entry["image"]).convert("RGB")).astype(float)
            mask = np.asarray(Image.open(entry["label"])) > 127
            if mask.sum() < 50:
                continue

            # Nuclei were rendered darker than their surroundings.
            assert image[mask].mean() < image[~mask].mean(), f"mask misaligned in {entry['image']}"
            checked += 1

        assert checked > 0, "no patch had enough foreground to check alignment"


class TestTrainAndInfer:
    @pytest.fixture(scope="class")
    def trained_checkpoint(self, prepared_dataset, bundle_root, tmp_path_factory) -> str:
        from monai.bundle import ConfigParser

        _, _, manifest_path = prepared_dataset
        ckpt_dir = str(tmp_path_factory.mktemp("e2e_models"))

        parser = ConfigParser()
        parser.read_config(os.path.join(bundle_root, "configs", "train.json"))
        parser.update({
            "bundle_root": bundle_root,
            "manifest": manifest_path,
            "ckpt_dir": ckpt_dir,
            "tb_dir": os.path.join(ckpt_dir, "tb"),
            "device": torch.device("cpu"),
            "max_epochs": 20,
            "batch_size": 4,
            "num_workers": 0,
            "learning_rate": 0.001,
            "amp": False,
            "patch_size": [128, 128],
        })
        for _ in parser.get_parsed_content("run"):
            pass

        checkpoint = os.path.join(ckpt_dir, "model.pt")
        assert os.path.exists(checkpoint), f"training produced no model.pt in {ckpt_dir}"
        return checkpoint

    def test_inferer_roi_matches_the_tile_size(self, bundle_root, trained_checkpoint):
        """Regression: the inferer must not be wider than the tiles it is fed.

        When roi_size (256, from the bundle default) exceeded the tile size
        (128), MONAI padded each tile and the network -- scoring 0.964 dice --
        emitted a maximum probability of 0.477, so thresholding at 0.5 yielded no
        objects at all. Nothing raised; inference just returned empty GeoJSON.
        """
        from tools.batch_infer import load_bundle_pieces

        pieces = load_bundle_pieces(bundle_root, trained_checkpoint, torch.device("cpu"), patch_size=128)
        assert max(int(r) for r in pieces["inferer"].roi_size) <= 128

    def test_confident_predictions_on_a_foreground_tile(self, bundle_root, trained_checkpoint,
                                                        prepared_dataset):
        """The model should cross the threshold somewhere on a nucleus-rich tile."""
        from PIL import Image

        from tools.batch_infer import load_bundle_pieces

        _, _, manifest_path = prepared_dataset
        with open(manifest_path, encoding="utf-8") as handle:
            entries = json.load(handle)["training"]

        entry = max(entries, key=lambda e: (np.asarray(Image.open(e["label"])) > 127).mean())
        patch = np.asarray(Image.open(entry["image"]).convert("RGB"))

        pieces = load_bundle_pieces(bundle_root, trained_checkpoint, torch.device("cpu"), patch_size=128)
        data = pieces["preprocessing"]({"image": np.moveaxis(patch, -1, 0).astype(np.float32)})
        tensor = torch.as_tensor(np.asarray(data["image"])).float().unsqueeze(0)

        with torch.no_grad():
            probability = torch.sigmoid(pieces["inferer"](tensor, pieces["network"]))

        assert float(probability.max()) > pieces["threshold"], (
            f"max probability {float(probability.max()):.3f} never reaches the "
            f"{pieces['threshold']} threshold, so inference can only return empty results"
        )

    def test_training_learns_something(self, trained_checkpoint):
        """Weights should exist and not be degenerate."""
        state = torch.load(trained_checkpoint, map_location="cpu", weights_only=False)
        tensors = [v for v in state.get("model", state).values() if torch.is_tensor(v) and v.is_floating_point()]
        assert tensors
        assert all(torch.isfinite(t).all() for t in tensors), "NaN/Inf in trained weights"

    def test_inference_recovers_the_ground_truth(self, prepared_dataset, trained_checkpoint,
                                                 bundle_root, tmp_path):
        """End of the loop: predictions must land on the real nuclei.

        The thresholds are loose on purpose -- this is eight CPU epochs on a toy
        slide, so the point is that the pipeline is wired correctly, not that the
        model is good.
        """
        from tools.batch_infer import infer_slide, load_bundle_pieces

        from monai_pathology.lib.utils.geojson_utils import (
            Region, annotations_to_mask, load_geojson, parse_annotations,
        )

        slide_path, truth_path, _ = prepared_dataset
        device = torch.device("cpu")

        pieces = load_bundle_pieces(bundle_root, trained_checkpoint, device, patch_size=128)
        output_path = infer_slide(
            slide_path=slide_path,
            pieces=pieces,
            output_dir=str(tmp_path),
            device=device,
            label="Nucleus",
            backend="TiffFile",
            patch_size=128,
            overlap=16,
            min_area=5.0,
            min_tissue_fraction=0.02,
            batch_size=4,
        )
        assert output_path and os.path.exists(output_path)

        truth = parse_annotations(load_geojson(truth_path))
        predicted = parse_annotations(load_geojson(output_path))
        assert predicted, "inference produced no objects at all"

        full = Region(0, 0, 1024, 1024, 1.0)
        truth_mask = annotations_to_mask(truth, full, {"Nucleus": 1}) > 0
        predicted_mask = annotations_to_mask(predicted, full, {"Nucleus": 1}) > 0

        intersection = np.logical_and(truth_mask, predicted_mask).sum()
        recall = intersection / truth_mask.sum()
        precision = intersection / max(predicted_mask.sum(), 1)

        assert recall > 0.5, f"recall {recall:.3f} -- predictions are missing the nuclei"
        assert precision > 0.5, f"precision {precision:.3f} -- predictions are mostly background"

    def test_predictions_are_valid_qupath_geojson(self, prepared_dataset, trained_checkpoint,
                                                  bundle_root, tmp_path):
        from tools.batch_infer import infer_slide, load_bundle_pieces

        slide_path, _, _ = prepared_dataset
        pieces = load_bundle_pieces(bundle_root, trained_checkpoint, torch.device("cpu"), patch_size=128)
        output_path = infer_slide(
            slide_path=slide_path, pieces=pieces, output_dir=str(tmp_path),
            device=torch.device("cpu"), label="Nucleus", backend="TiffFile",
            patch_size=128, overlap=0, min_tissue_fraction=0.05, batch_size=4,
        )

        with open(output_path, encoding="utf-8") as handle:
            geojson = json.load(handle)

        assert geojson["type"] == "FeatureCollection"
        for feature in geojson["features"][:20]:
            assert feature["geometry"]["type"] == "Polygon"
            ring = feature["geometry"]["coordinates"][0]
            assert len(ring) >= 4 and ring[0] == ring[-1], "GeoJSON rings must be closed"
            assert feature["properties"]["classification"]["name"] == "Nucleus"
