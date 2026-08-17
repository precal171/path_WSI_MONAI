#!/usr/bin/env python3
"""Generate a fake pyramidal slide plus matching ground-truth GeoJSON.

The point is to give the whole pipeline something to chew on before you have
real annotated data, or a GPU, or a working QuPath connection. Everything
downstream -- ``prepare_training_data.py``, bundle training, ``batch_infer.py``,
the GeoJSON round trip -- can be exercised end to end against the output of this
script, which is also what the test suite uses.

The "tissue" is crude on purpose: pinkish stroma with darker blue-purple blobs
standing in for nuclei. It is not meant to look convincing, only to have the
right *shape*: RGB, pyramidal, tiled, with sparse foreground on a bright
background and a known set of object outlines to check predictions against.

Usage:
    python tools/build_synthetic_wsi.py --output-dir data/synthetic
    python tools/build_synthetic_wsi.py --output-dir data/synthetic \\
        --width 8192 --height 8192 --num-nuclei 3000 --seed 7

Writes:
    <output-dir>/synthetic_slide.tiff          pyramidal tiled RGB TIFF
    <output-dir>/synthetic_slide.geojson       ground-truth nucleus outlines
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys

import numpy as np

logger = logging.getLogger("build_synthetic_wsi")


def _ellipse_polygon(cx: float, cy: float, rx: float, ry: float, angle: float, n: int = 16) -> list[list[float]]:
    """Vertices of a rotated ellipse, closed, as ``[x, y]`` pairs."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    points = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        dx, dy = rx * math.cos(t), ry * math.sin(t)
        points.append([cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a])
    points.append(points[0])
    return points


def build_slide(
    width: int = 4096,
    height: int = 4096,
    num_nuclei: int = 800,
    tissue_fraction: float = 0.6,
    seed: int = 0,
) -> tuple[np.ndarray, list[list[list[float]]]]:
    """Render the slide image and return it with the nucleus polygons.

    Returns:
        ``(rgb, polygons)`` where ``rgb`` is ``(H, W, 3)`` uint8 and each polygon
        is a list of closed ``[x, y]`` level-0 vertices.
    """
    rng = np.random.default_rng(seed)

    # Bright, slightly noisy glass background.
    image = rng.normal(planned := 238, 4, (height, width, 3)).clip(200, 255).astype(np.uint8)
    del planned

    # A blobby tissue area so the slide is not uniformly covered -- this is what
    # makes tissue_mask() and foreground filtering meaningful.
    yy, xx = np.mgrid[0:height, 0:width]
    tissue = np.zeros((height, width), dtype=bool)
    for _ in range(6):
        cy = rng.uniform(0.2, 0.8) * height
        cx = rng.uniform(0.2, 0.8) * width
        radius = rng.uniform(0.2, 0.45) * min(width, height) * tissue_fraction
        tissue |= ((yy - cy) ** 2 + (xx - cx) ** 2) < radius**2

    # Eosin-ish pink for stroma.
    stroma = rng.normal([225, 190, 215], 6, (height, width, 3)).clip(0, 255).astype(np.uint8)
    image[tissue] = stroma[tissue]

    # Haematoxylin-ish nuclei, only inside tissue.
    tissue_indices = np.argwhere(tissue)
    if len(tissue_indices) == 0:
        raise RuntimeError("No tissue was generated; try a larger tissue_fraction.")

    polygons: list[list[list[float]]] = []
    for _ in range(num_nuclei):
        cy, cx = tissue_indices[rng.integers(len(tissue_indices))]
        rx, ry = rng.uniform(6, 13), rng.uniform(6, 13)
        angle = rng.uniform(0, math.pi)

        # Rasterise into a local bounding box rather than the whole slide --
        # doing this full-frame per nucleus is what makes naive versions of this
        # script take minutes instead of seconds.
        pad = int(max(rx, ry)) + 2
        y0, y1 = max(0, cy - pad), min(height, cy + pad + 1)
        x0, x1 = max(0, cx - pad), min(width, cx + pad + 1)
        if y1 - y0 < 3 or x1 - x0 < 3:
            continue

        ly, lx = np.mgrid[y0:y1, x0:x1]
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        dx, dy = lx - cx, ly - cy
        rot_x, rot_y = dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a
        blob = (rot_x / rx) ** 2 + (rot_y / ry) ** 2 <= 1.0
        if not blob.any():
            continue

        colour = rng.normal([95, 70, 150], 10, 3).clip(0, 255).astype(np.uint8)
        patch = image[y0:y1, x0:x1]
        patch[blob] = colour
        image[y0:y1, x0:x1] = patch

        polygons.append(_ellipse_polygon(float(cx), float(cy), rx, ry, angle))

    logger.info("Rendered %d nuclei over %.1f%% tissue", len(polygons), 100 * tissue.mean())
    return image, polygons


def write_pyramid(image: np.ndarray, path: str, tile_size: int = 256, num_levels: int = 4) -> None:
    """Write a tiled, pyramidal TIFF.

    Written with SubIFDs so the resolution levels form a proper pyramid, which
    is what lets ``WSIReader`` report more than one level. Note that OpenSlide
    only recognises pyramids in vendor formats (SVS, NDPI, ...), so read this
    file with the ``TiffFile`` backend -- see ``tests/test_wsi_patch_extraction.py``.
    """
    import tifffile

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    levels = [image]
    for _ in range(1, num_levels):
        previous = levels[-1]
        if min(previous.shape[:2]) < tile_size * 2:
            break
        levels.append(previous[::2, ::2])

    # JPEG keeps the fixture small, but it needs imagecodecs. Uncompressed is a
    # perfectly good fallback for a test fixture -- just bigger.
    try:
        import imagecodecs  # noqa: F401

        compression: str | None = "jpeg"
    except ImportError:
        logger.warning("imagecodecs not installed; writing the TIFF uncompressed.")
        compression = None

    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        writer.write(
            levels[0],
            tile=(tile_size, tile_size),
            photometric="rgb",
            compression=compression,
            subifds=len(levels) - 1,
            metadata={"axes": "YXS"},
        )
        for level in levels[1:]:
            writer.write(
                level,
                tile=(tile_size, tile_size),
                photometric="rgb",
                compression=compression,
                subfiletype=1,
                metadata={"axes": "YXS"},
            )

    logger.info("Wrote %s with %d levels (%s)", path, len(levels), [l.shape[:2] for l in levels])


def write_geojson(polygons: list[list[list[float]]], path: str, label: str = "Nucleus") -> None:
    """Write the ground-truth outlines in the same GeoJSON dialect QuPath emits."""
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [polygon]},
            "properties": {
                "objectType": "annotation",
                "classification": {"name": label, "color": [200, 0, 200]},
            },
        }
        for polygon in polygons
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features}, handle)
    logger.info("Wrote %s with %d annotations", path, len(features))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/synthetic", help="where to write the slide and GeoJSON")
    parser.add_argument("--name", default="synthetic_slide", help="basename for the outputs")
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--height", type=int, default=4096)
    parser.add_argument("--num-nuclei", type=int, default=800)
    parser.add_argument("--tissue-fraction", type=float, default=0.6)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--num-levels", type=int, default=4)
    parser.add_argument("--label", default="Nucleus", help="class name for the ground-truth annotations")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    image, polygons = build_slide(
        width=args.width,
        height=args.height,
        num_nuclei=args.num_nuclei,
        tissue_fraction=args.tissue_fraction,
        seed=args.seed,
    )

    slide_path = os.path.join(args.output_dir, f"{args.name}.tiff")
    geojson_path = os.path.join(args.output_dir, f"{args.name}.geojson")
    write_pyramid(image, slide_path, tile_size=args.tile_size, num_levels=args.num_levels)
    write_geojson(polygons, geojson_path, label=args.label)

    print(f"\nSlide:      {slide_path}")
    print(f"Annotations:{geojson_path}")
    print("\nNext:  python tools/prepare_training_data.py \\")
    print(f"           --slide {slide_path} --annotations {geojson_path} \\")
    print("           --output-dir data/patches --labels Nucleus")
    return 0


if __name__ == "__main__":
    sys.exit(main())
