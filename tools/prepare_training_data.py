#!/usr/bin/env python3
"""Turn QuPath annotations into a patch dataset the bundle can train on.

Reads one or more slides plus their GeoJSON annotations, cuts patches, rasterises
the matching label masks, and writes a ``manifest.json`` that
``bundles/*/configs/train.json`` consumes directly.

A note on annotation completeness, because it determines whether the resulting
model is any good:

    A patch is only valid training data if **everything** in it was annotated.
    If you outlined twelve nuclei in a field that contains forty, the other
    twenty-eight become background in the mask, and the model is actively taught
    that nuclei are background.

So by default this script only emits patches that overlap an annotation, which
is a rough proxy for "the pathologist was looking here". If you draw explicit
fully-annotated boxes in QuPath -- give them a class like ``ROI`` -- pass
``--roi-labels ROI`` and only those areas are tiled. That is the honest way to
do it, and is strongly recommended once you move past a first smoke test.

Usage:
    # single slide
    python tools/prepare_training_data.py \\
        --slide data/synthetic/synthetic_slide.tiff \\
        --annotations data/synthetic/synthetic_slide.geojson \\
        --output-dir data/patches --labels Nucleus --backend TiffFile

    # a directory of slides, paired with .geojson files by basename
    python tools/prepare_training_data.py \\
        --input-dir /archive/slides --output-dir data/patches \\
        --labels Nucleus --roi-labels ROI
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai_pathology.lib.utils.geojson_utils import (  # noqa: E402
    Annotation,
    Region,
    annotations_to_mask,
    load_geojson,
    parse_annotations,
)
from monai_pathology.lib.utils.wsi_utils import (  # noqa: E402
    DEFAULT_BACKEND,
    get_slide_info,
    open_slide,
    read_region,
    tile_regions,
)

logger = logging.getLogger("prepare_training_data")

SLIDE_EXTENSIONS = (".svs", ".ndpi", ".tiff", ".tif", ".scn", ".mrxs", ".vms", ".bif", ".czi")


def _bounds_of(annotations: list[Annotation]) -> tuple[int, int, int, int] | None:
    """Level-0 ``(x0, y0, x1, y1)`` bounding box covering every annotation."""
    if not annotations:
        return None
    xs0, ys0, xs1, ys1 = [], [], [], []
    for ann in annotations:
        xs0.append(ann.exterior[:, 0].min())
        ys0.append(ann.exterior[:, 1].min())
        xs1.append(ann.exterior[:, 0].max())
        ys1.append(ann.exterior[:, 1].max())
    return int(min(xs0)), int(min(ys0)), int(max(xs1)), int(max(ys1))


def _tiling_areas(
    positives: list[Annotation],
    rois: list[Annotation],
    slide_size: tuple[int, int],
    patch_size: int,
    stride: int,
    downsample: float,
) -> list[Region]:
    """Decide which parts of the slide to cut patches from."""
    areas: list[Region] = []

    if rois:
        # Explicit fully-annotated regions: tile inside each one.
        for roi in rois:
            x0, y0 = roi.exterior[:, 0].min(), roi.exterior[:, 1].min()
            x1, y1 = roi.exterior[:, 0].max(), roi.exterior[:, 1].max()
            bounds = Region(int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0)), downsample)
            areas.extend(tile_regions(slide_size, patch_size, stride, downsample, bounds))
        return areas

    box = _bounds_of(positives)
    if box is None:
        return []
    x0, y0, x1, y1 = box
    # Pad by one patch so annotations near the edge still land in a full tile.
    pad = int(patch_size * downsample)
    bounds = Region(
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(slide_size[0], x1 + pad) - max(0, x0 - pad),
        min(slide_size[1], y1 + pad) - max(0, y0 - pad),
        downsample,
    )
    return list(tile_regions(slide_size, patch_size, stride, downsample, bounds))


def process_slide(
    slide_path: str,
    annotation_path: str,
    output_dir: str,
    labels: list[str],
    roi_labels: list[str],
    patch_size: int = 256,
    stride: int | None = None,
    downsample: float = 1.0,
    min_positive_fraction: float = 0.0,
    background_fraction: float = 0.0,
    backend: str = DEFAULT_BACKEND,
    seed: int = 0,
) -> list[dict[str, str]]:
    """Cut one slide into patch/mask pairs. Returns manifest entries."""
    stride = stride or patch_size
    rng = np.random.default_rng(seed)

    geojson = load_geojson(annotation_path)
    positives = parse_annotations(geojson, keep_labels=labels)
    rois = parse_annotations(geojson, keep_labels=roi_labels) if roi_labels else []
    logger.info(
        "%s: %d annotations across %s%s",
        os.path.basename(slide_path),
        len(positives),
        labels,
        f", {len(rois)} ROI(s)" if rois else "",
    )
    if not positives:
        logger.warning("%s has no annotations matching %s -- skipping.", slide_path, labels)
        return []

    label_map = {name: i + 1 for i, name in enumerate(labels)}
    info = get_slide_info(slide_path, backend=backend)
    reader, wsi = open_slide(slide_path, backend=backend)

    stem = os.path.splitext(os.path.basename(slide_path))[0]
    image_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    from PIL import Image

    entries: list[dict[str, str]] = []
    kept_background = skipped = 0

    for region in _tiling_areas(positives, rois, info.size, patch_size, stride, downsample):
        mask = annotations_to_mask(positives, region, label_map=label_map)
        positive_fraction = float((mask > 0).mean())

        if positive_fraction <= 0.0:
            # An empty tile. Keep a controlled share so the model sees true
            # negatives, but not so many that the loss is dominated by them.
            if background_fraction <= 0 or rng.random() > background_fraction:
                skipped += 1
                continue
            kept_background += 1
        elif positive_fraction < min_positive_fraction:
            skipped += 1
            continue

        patch = read_region(reader, wsi, region)
        if patch.ndim == 3 and patch.shape[2] > 3:
            patch = patch[..., :3]  # drop alpha if the reader supplied one

        name = f"{stem}_x{region.x}_y{region.y}_d{int(region.downsample)}"
        image_path = os.path.join(image_dir, f"{name}.png")
        mask_path = os.path.join(mask_dir, f"{name}.png")

        Image.fromarray(patch.astype(np.uint8)).save(image_path)
        # Scale to 0/255 so masks are viewable; train.json rescales them back.
        Image.fromarray((mask > 0).astype(np.uint8) * 255).save(mask_path)

        entries.append({"image": os.path.abspath(image_path), "label": os.path.abspath(mask_path), "slide": stem})

    logger.info(
        "%s: wrote %d patches (%d background, %d skipped)",
        os.path.basename(slide_path), len(entries), kept_background, skipped,
    )
    return entries


def _pair_inputs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Resolve --slide/--annotations pairs, or pair up a directory by basename."""
    if args.slide:
        if len(args.slide) != len(args.annotations):
            raise SystemExit(
                f"Got {len(args.slide)} --slide and {len(args.annotations)} --annotations; "
                "they must come in pairs."
            )
        return list(zip(args.slide, args.annotations))

    pairs: list[tuple[str, str]] = []
    for entry in sorted(os.listdir(args.input_dir)):
        if not entry.lower().endswith(SLIDE_EXTENSIONS):
            continue
        slide = os.path.join(args.input_dir, entry)
        geojson = os.path.join(args.input_dir, os.path.splitext(entry)[0] + ".geojson")
        if os.path.exists(geojson):
            pairs.append((slide, geojson))
        else:
            logger.warning("No .geojson alongside %s -- skipping.", entry)
    if not pairs:
        raise SystemExit(f"No slide/geojson pairs found in {args.input_dir}")
    return pairs


def _split(entries: list[dict[str, str]], val_fraction: float, seed: int) -> tuple[list, list]:
    """Hold out a validation set, by slide when possible.

    Splitting patches from one slide across train and validation leaks: adjacent
    tiles share tissue, staining and scanner characteristics, so validation
    scores come out optimistic. With more than one slide we split by slide
    instead, which is the honest measure of whether the model generalises.
    """
    rng = np.random.default_rng(seed)
    slides = sorted({e["slide"] for e in entries})

    if len(slides) > 1:
        shuffled = list(rng.permutation(slides))
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        val_slides = set(shuffled[:n_val])
        train = [e for e in entries if e["slide"] not in val_slides]
        val = [e for e in entries if e["slide"] in val_slides]
        logger.info("Split by slide: %d train / %d val (val slides: %s)", len(train), len(val), sorted(val_slides))
        return train, val

    logger.warning(
        "Only one slide, so train/validation are split by patch. Validation scores "
        "will be optimistic -- add more slides before trusting them."
    )
    order = rng.permutation(len(entries))
    n_val = max(1, int(round(len(entries) * val_fraction))) if entries else 0
    val_idx = set(order[:n_val].tolist())
    train = [e for i, e in enumerate(entries) if i not in val_idx]
    val = [e for i, e in enumerate(entries) if i in val_idx]
    return train, val


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--slide", action="append", help="path to a slide (repeatable, pairs with --annotations)")
    source.add_argument("--input-dir", help="directory of slides, each with a matching .geojson")
    parser.add_argument("--annotations", action="append", default=[], help="GeoJSON for the matching --slide")
    parser.add_argument("--output-dir", required=True, help="where patches and manifest.json are written")
    parser.add_argument("--labels", nargs="+", required=True, help="annotation classes to train on, e.g. Nucleus")
    parser.add_argument(
        "--roi-labels", nargs="*", default=[],
        help="classes marking fully-annotated regions. Strongly recommended -- see the module docstring.",
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None, help="defaults to --patch-size (no overlap)")
    parser.add_argument("--downsample", type=float, default=1.0, help="level-0 pixels per patch pixel")
    parser.add_argument("--min-positive-fraction", type=float, default=0.0,
                        help="drop patches with less foreground than this")
    parser.add_argument("--background-fraction", type=float, default=0.1,
                        help="probability of keeping an empty patch as a true negative")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help="WSI backend: OpenSlide (default), TiffFile, or cuCIM")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    entries: list[dict[str, str]] = []
    for slide_path, annotation_path in _pair_inputs(args):
        entries.extend(
            process_slide(
                slide_path=slide_path,
                annotation_path=annotation_path,
                output_dir=args.output_dir,
                labels=args.labels,
                roi_labels=args.roi_labels,
                patch_size=args.patch_size,
                stride=args.stride,
                downsample=args.downsample,
                min_positive_fraction=args.min_positive_fraction,
                background_fraction=args.background_fraction,
                backend=args.backend,
                seed=args.seed,
            )
        )

    if not entries:
        raise SystemExit("No patches were produced. Check --labels against the classes in your GeoJSON.")

    train, val = _split(entries, args.val_fraction, args.seed)
    manifest = {
        "task": "nuclei_segmentation",
        "created": datetime.now(timezone.utc).isoformat(),
        "labels": args.labels,
        "patch_size": [args.patch_size, args.patch_size],
        "downsample": args.downsample,
        "num_patches": len(entries),
        "training": train,
        "validation": val,
    }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\n{len(entries)} patches -> {args.output_dir}")
    print(f"  training:   {len(train)}")
    print(f"  validation: {len(val)}")
    print(f"  manifest:   {manifest_path}")
    print("\nNext:  python -m monai.bundle run \\")
    print("           --config_file bundles/nuclei_segmentation/configs/train.json \\")
    print("           --bundle_root bundles/nuclei_segmentation \\")
    print(f"           --manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
