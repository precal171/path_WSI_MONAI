#!/usr/bin/env python3
"""Apply a trained bundle across whole slides, writing GeoJSON for QuPath.

This is the non-interactive half of the loop: no MONAI Label server, no QuPath
connection, just a SLURM job chewing through slides. Import the results into
QuPath (``Objects > Import objects``) to review them, correct what is wrong, and
feed the corrections back as new training data.

The transform chain, network and inferer are all pulled out of the bundle's
``inference.json`` rather than redefined here, so batch inference cannot drift
away from what the model was trained on. Only the WSI tiling and the
mask-to-GeoJSON step are this script's own.

Usage:
    python tools/batch_infer.py \\
        --bundle bundles/nuclei_segmentation \\
        --slide data/synthetic/synthetic_slide.tiff \\
        --output-dir outputs --backend TiffFile

    # a whole directory, as run from slurm/infer_wsi_batch.sbatch
    python tools/batch_infer.py --bundle bundles/nuclei_segmentation \\
        --input-dir /archive/slides --output-dir outputs
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai_pathology.lib.utils.geojson_utils import (  # noqa: E402
    Annotation,
    Region,
    annotations_to_geojson,
    load_geojson,
    mask_to_annotations,
    parse_annotations,
    save_geojson,
)
from monai_pathology.lib.utils.wsi_utils import (  # noqa: E402
    DEFAULT_BACKEND,
    get_slide_info,
    open_slide,
    read_region,
    tile_regions,
    tissue_mask,
)

logger = logging.getLogger("batch_infer")

SLIDE_EXTENSIONS = (".svs", ".ndpi", ".tiff", ".tif", ".scn", ".mrxs", ".vms", ".bif", ".czi")


def load_bundle_pieces(
    bundle_root: str,
    ckpt_path: str | None,
    device: torch.device,
    patch_size: int | None = None,
):
    """Resolve network, preprocessing, inferer and postprocessing from the bundle.

    Args:
        patch_size: the tile size this run will actually feed the model. It is
            pushed into the config so the SlidingWindowInferer's ``roi_size``
            matches. Leaving them out of sync is quietly destructive: if
            ``roi_size`` is larger than the tile, MONAI pads the tile out to
            ``roi_size`` and the network sees a border of constant padding it
            never saw in training. Predictions come back systematically
            under-confident -- in testing, a model scoring 0.96 dice produced a
            maximum probability of 0.477 and therefore *zero* objects after
            thresholding, with no error anywhere.
    """
    from monai.bundle import ConfigParser

    # bundle_root must be importable for `$import scripts` inside the config.
    if bundle_root not in sys.path:
        sys.path.insert(0, bundle_root)

    config_path = os.path.join(bundle_root, "configs", "inference.json")
    parser = ConfigParser()
    parser.read_config(config_path)
    parser.update({"bundle_root": bundle_root, "device": device})
    if patch_size:
        parser.update({"patch_size": [int(patch_size), int(patch_size)]})
    if ckpt_path:
        parser.update({"ckpt_path": ckpt_path})

    network = parser.get_parsed_content("network")
    resolved_ckpt = parser.get_parsed_content("ckpt_path")

    from scripts.utils import load_weights  # bundle-local helper

    load_weights(network, resolved_ckpt, device)
    network.eval()

    return {
        "network": network,
        # The array variant, not @preprocessing: patches arrive from the WSI
        # reader as arrays, so the LoadImaged step must not be in the chain.
        "preprocessing": parser.get_parsed_content("array_preprocessing"),
        "inferer": parser.get_parsed_content("inferer"),
        "threshold": float(parser.get_parsed_content("threshold")),
        "patch_size": list(parser.get_parsed_content("patch_size")),
    }


def _foreground_regions(
    slide_path: str,
    backend: str,
    patch_size: int,
    stride: int,
    downsample: float,
    roi_path: str | None,
    tissue_downsample: float,
    min_tissue_fraction: float,
) -> list[Region]:
    """Tiles worth running the model on: inside any ROI, and containing tissue."""
    info = get_slide_info(slide_path, backend=backend)

    bounds_list: list[Region | None] = [None]
    if roi_path:
        rois = parse_annotations(load_geojson(roi_path))
        if rois:
            bounds_list = []
            for roi in rois:
                x0, y0 = roi.exterior[:, 0].min(), roi.exterior[:, 1].min()
                x1, y1 = roi.exterior[:, 0].max(), roi.exterior[:, 1].max()
                bounds_list.append(
                    Region(int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0)), downsample)
                )
            logger.info("Restricting to %d ROI(s) from %s", len(bounds_list), os.path.basename(roi_path))

    candidates: list[Region] = []
    for bounds in bounds_list:
        candidates.extend(tile_regions(info.size, patch_size, stride, downsample, bounds))

    if min_tissue_fraction <= 0:
        return candidates

    # One cheap low-resolution pass tells us which tiles are glass. Skipping
    # those is the difference between minutes and hours on a real slide, which
    # is mostly empty.
    reader, wsi = open_slide(slide_path, backend=backend)
    thumb_region = Region(0, 0, info.size[0], info.size[1], tissue_downsample)
    thumbnail = read_region(reader, wsi, thumb_region)
    mask = tissue_mask(thumbnail)

    kept: list[Region] = []
    for region in candidates:
        r0 = int(region.y / tissue_downsample)
        r1 = int((region.y + region.height) / tissue_downsample)
        c0 = int(region.x / tissue_downsample)
        c1 = int((region.x + region.width) / tissue_downsample)
        window = mask[max(0, r0):max(r0 + 1, r1), max(0, c0):max(c0 + 1, c1)]
        if window.size and window.mean() >= min_tissue_fraction:
            kept.append(region)

    logger.info(
        "%s: %d/%d tiles contain tissue (>=%.0f%%)",
        os.path.basename(slide_path), len(kept), len(candidates), 100 * min_tissue_fraction,
    )
    return kept


def _merge_split_objects(annotations: list[Annotation], label: str | None) -> list[Annotation]:
    """Rejoin objects cut in half by a tile boundary.

    Only *overlapping* polygons are merged. Two distinct nuclei that happen to
    touch normally abut rather than overlap, whereas the two halves of one
    nucleus straddling a tile edge do overlap once the tiles overlap. That makes
    intersection a reasonable discriminator -- but it is a heuristic, and with
    densely packed nuclei it will occasionally fuse two real objects.
    """
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        logger.warning("shapely is not installed; leaving boundary-split objects as they are.")
        return annotations

    polygons = []
    for ann in annotations:
        try:
            poly = Polygon(ann.exterior, [r for r in ann.interiors if len(r) >= 3])
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)
        except Exception:  # noqa: BLE001 - a malformed contour should not sink the run
            continue

    if not polygons:
        return annotations

    merged = unary_union(polygons)
    geoms = list(getattr(merged, "geoms", [merged]))

    out: list[Annotation] = []
    for geom in geoms:
        if geom.geom_type != "Polygon" or geom.is_empty:
            continue
        out.append(
            Annotation(
                exterior=np.asarray(geom.exterior.coords, dtype=np.float64),
                interiors=[np.asarray(r.coords, dtype=np.float64) for r in geom.interiors],
                label=label,
                properties={"objectType": "annotation"},
            )
        )
    logger.info("Merged %d contours into %d objects", len(polygons), len(out))
    return out


def infer_slide(
    slide_path: str,
    pieces: dict,
    output_dir: str,
    device: torch.device,
    label: str = "Nucleus",
    backend: str = DEFAULT_BACKEND,
    patch_size: int = 512,
    overlap: int = 64,
    downsample: float = 1.0,
    min_area: float = 10.0,
    simplify_tolerance: float = 1.0,
    min_tissue_fraction: float = 0.05,
    tissue_downsample: float = 32.0,
    roi_path: str | None = None,
    merge_boundaries: bool = True,
    object_type: str = "detection",
    batch_size: int = 4,
) -> str:
    """Run the model over one slide and write its GeoJSON. Returns the path."""
    started = time.time()
    stride = max(1, patch_size - overlap)

    regions = _foreground_regions(
        slide_path, backend, patch_size, stride, downsample,
        roi_path, tissue_downsample, min_tissue_fraction,
    )
    if not regions:
        logger.warning("%s: nothing to do.", slide_path)
        return ""

    reader, wsi = open_slide(slide_path, backend=backend)
    preprocessing = pieces["preprocessing"]
    network = pieces["network"]
    inferer = pieces["inferer"]
    threshold = pieces["threshold"]

    # Belt and braces: load_bundle_pieces should already have matched these, but
    # a mismatch here degrades every prediction silently rather than failing, so
    # it is worth catching however the pieces were assembled.
    roi = getattr(inferer, "roi_size", None)
    if roi is not None and max(int(r) for r in roi) > patch_size:
        logger.warning(
            "Inferer roi_size %s exceeds the tile size %d; tiles will be padded and "
            "predictions will be under-confident. Pass patch_size to load_bundle_pieces.",
            list(roi), patch_size,
        )

    annotations: list[Annotation] = []
    buffer: list[tuple[Region, torch.Tensor]] = []

    def flush() -> None:
        if not buffer:
            return
        batch = torch.stack([t for _, t in buffer]).to(device)
        with torch.no_grad():
            logits = inferer(batch, network)
            probabilities = torch.sigmoid(logits)
        binary = (probabilities >= threshold).to(torch.uint8).cpu().numpy()

        for (region, _), prediction in zip(buffer, binary):
            mask = prediction[0] if prediction.ndim == 3 else prediction
            if not mask.any():
                continue
            annotations.extend(
                mask_to_annotations(
                    mask, region,
                    label_map={label: 1},
                    separate_instances=True,
                    min_area=min_area,
                    simplify_tolerance=simplify_tolerance,
                )
            )
        buffer.clear()

    for index, region in enumerate(regions):
        patch = read_region(reader, wsi, region)
        if patch.ndim == 3 and patch.shape[2] > 3:
            patch = patch[..., :3]

        # Route through the bundle's own preprocessing so scaling matches training.
        data = preprocessing({"image": np.moveaxis(patch, -1, 0).astype(np.float32)})
        tensor = data["image"] if isinstance(data, dict) else data
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(np.asarray(tensor))
        buffer.append((region, tensor.float()))

        if len(buffer) >= batch_size:
            flush()
        if (index + 1) % 50 == 0:
            logger.info("  %s: %d/%d tiles", os.path.basename(slide_path), index + 1, len(regions))

    flush()

    if merge_boundaries and overlap > 0:
        annotations = _merge_split_objects(annotations, label)

    geojson = annotations_to_geojson(annotations, colors={label: [200, 0, 200]}, object_type=object_type)
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(slide_path))[0]
    output_path = os.path.join(output_dir, f"{stem}.geojson")
    save_geojson(geojson, output_path)

    logger.info(
        "%s: %d objects -> %s (%.1fs)",
        os.path.basename(slide_path), len(annotations), output_path, time.time() - started,
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--slide", action="append", help="slide to process (repeatable)")
    source.add_argument("--input-dir", help="directory of slides")
    parser.add_argument("--bundle", required=True, help="bundle root, e.g. bundles/nuclei_segmentation")
    parser.add_argument("--ckpt", default=None, help="checkpoint; defaults to <bundle>/models/model.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="Nucleus", help="class name written into the GeoJSON")
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64, help="tile overlap in patch pixels; 0 disables merging")
    parser.add_argument("--downsample", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-area", type=float, default=10.0, help="drop objects smaller than this many pixels")
    parser.add_argument("--simplify", type=float, default=1.0, help="polygon simplification tolerance, 0 disables")
    parser.add_argument("--min-tissue-fraction", type=float, default=0.05)
    parser.add_argument("--tissue-downsample", type=float, default=32.0)
    parser.add_argument("--roi", default=None, help="GeoJSON restricting inference to certain regions")
    parser.add_argument("--no-merge", action="store_true", help="keep boundary-split objects separate")
    parser.add_argument("--object-type", default="detection", choices=["detection", "annotation"],
                        help="QuPath object type; 'detection' is lighter for large counts")
    parser.add_argument("--shard-index", type=int, default=0, help="for SLURM job arrays")
    parser.add_argument("--num-shards", type=int, default=1, help="for SLURM job arrays")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.slide:
        slides = args.slide
    else:
        slides = sorted(
            p for p in glob.glob(os.path.join(args.input_dir, "*"))
            if p.lower().endswith(SLIDE_EXTENSIONS)
        )
    if not slides:
        raise SystemExit("No slides found.")

    if args.num_shards > 1:
        slides = slides[args.shard_index :: args.num_shards]
        logger.info("Shard %d/%d: %d slides", args.shard_index, args.num_shards, len(slides))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running on %s", device)

    pieces = load_bundle_pieces(os.path.abspath(args.bundle), args.ckpt, device, patch_size=args.patch_size)

    for slide in slides:
        try:
            infer_slide(
                slide_path=slide,
                pieces=pieces,
                output_dir=args.output_dir,
                device=device,
                label=args.label,
                backend=args.backend,
                patch_size=args.patch_size,
                overlap=args.overlap,
                downsample=args.downsample,
                min_area=args.min_area,
                simplify_tolerance=args.simplify,
                min_tissue_fraction=args.min_tissue_fraction,
                tissue_downsample=args.tissue_downsample,
                roi_path=args.roi,
                merge_boundaries=not args.no_merge,
                object_type=args.object_type,
                batch_size=args.batch_size,
            )
        except Exception:  # noqa: BLE001 - one bad slide must not kill a long batch job
            logger.exception("Failed on %s; continuing.", slide)

    return 0


if __name__ == "__main__":
    sys.exit(main())
