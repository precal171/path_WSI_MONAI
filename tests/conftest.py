"""Shared fixtures.

Everything here is synthetic. No test in this suite needs a real slide, a GPU, a
QuPath install or a running MONAI Label server -- see docs/testing.md for what
that does and does not prove.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_ROOT = os.path.join(REPO_ROOT, "bundles", "nuclei_segmentation")

for path in (REPO_ROOT, BUNDLE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def repo_root() -> str:
    return REPO_ROOT


@pytest.fixture(scope="session")
def bundle_root() -> str:
    return BUNDLE_ROOT


@pytest.fixture(scope="session")
def synthetic_slide(tmp_path_factory) -> tuple[str, str]:
    """A small pyramidal TIFF plus its ground-truth GeoJSON.

    Session-scoped: generating it costs a second or two, and every test that
    wants it only reads it.
    """
    from tools.build_synthetic_wsi import build_slide, write_geojson, write_pyramid

    directory = tmp_path_factory.mktemp("wsi")
    slide_path = str(directory / "slide.tiff")
    geojson_path = str(directory / "slide.geojson")

    image, polygons = build_slide(width=1024, height=1024, num_nuclei=120, seed=42)
    write_pyramid(image, slide_path, tile_size=256, num_levels=2)
    write_geojson(polygons, geojson_path)

    return slide_path, geojson_path


@pytest.fixture(scope="session")
def patch_manifest(tmp_path_factory) -> str:
    """A tiny image/mask patch dataset with a manifest, as train.json expects."""
    directory = tmp_path_factory.mktemp("patches")
    rng = np.random.default_rng(0)
    entries = []

    for index in range(6):
        image = rng.integers(180, 240, (128, 128, 3), dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        rows, cols = np.mgrid[0:128, 0:128]
        for _ in range(8):
            cy, cx, radius = rng.integers(15, 113), rng.integers(15, 113), rng.integers(5, 11)
            blob = (rows - cy) ** 2 + (cols - cx) ** 2 < radius**2
            mask[blob] = 255
            image[blob] = rng.integers(60, 110, 3, dtype=np.uint8)

        image_path = str(directory / f"img_{index}.png")
        mask_path = str(directory / f"msk_{index}.png")
        Image.fromarray(image).save(image_path)
        Image.fromarray(mask).save(mask_path)
        entries.append({"image": image_path, "label": mask_path, "slide": "synthetic"})

    manifest_path = str(directory / "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "nuclei_segmentation",
                "labels": ["Nucleus"],
                "patch_size": [128, 128],
                "training": entries[:4],
                "validation": entries[4:],
            },
            handle,
        )
    return manifest_path
