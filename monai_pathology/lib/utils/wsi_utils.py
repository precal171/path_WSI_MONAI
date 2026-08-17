"""Whole-slide reading helpers built on :class:`monai.data.WSIReader`.

Two things this module exists to absorb:

1. **Coordinate order.** ``WSIReader.get_data`` takes ``location=(row, col)`` and
   ``size=(height, width)`` -- ``(y, x)`` order -- while QuPath GeoJSON, and
   therefore :class:`~monai_pathology.lib.utils.geojson_utils.Region`, use
   ``(x, y)``. Every call here goes through :func:`read_region`, which takes a
   ``Region`` and does the flip once.

2. **Backend defaults.** ``WSIReader`` defaults to the ``cucim`` backend, which
   is usually not what you want on a shared cluster (see requirements-train.txt
   for why). Everything here defaults to OpenSlide instead.

Also note that ``location`` is always in **level-0** coordinates even when you
are reading at a higher level, whereas ``size`` is in pixels **at the level you
asked for**. That asymmetry is a common source of misaligned patches;
:func:`read_region` derives both from a single ``Region`` so callers cannot mix
them up.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterator, Sequence

import numpy as np

from .geojson_utils import Region

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BACKEND",
    "SlideInfo",
    "open_slide",
    "get_slide_info",
    "read_region",
    "tissue_mask",
    "tile_regions",
]

#: OpenSlide is the default everywhere in this project. Override per call if your
#: cluster has a working cuCIM build.
DEFAULT_BACKEND = "OpenSlide"


class SlideInfo:
    """Geometry of a slide: level dimensions, downsamples, and resolution."""

    def __init__(
        self,
        path: str,
        level_dimensions: Sequence[tuple[int, int]],
        level_downsamples: Sequence[float],
        mpp: tuple[float, float] | None = None,
        objective_power: float | None = None,
    ) -> None:
        self.path = path
        self.level_dimensions = [tuple(d) for d in level_dimensions]
        self.level_downsamples = [float(d) for d in level_downsamples]
        self.mpp = mpp
        self.objective_power = objective_power

    @property
    def num_levels(self) -> int:
        return len(self.level_dimensions)

    @property
    def size(self) -> tuple[int, int]:
        """Level-0 size as ``(width, height)``."""
        return self.level_dimensions[0]

    def level_for_downsample(self, downsample: float) -> int:
        """Index of the level whose downsample is closest to (but not past) ``downsample``."""
        usable = [i for i, d in enumerate(self.level_downsamples) if d <= downsample * 1.01]
        return max(usable, key=lambda i: self.level_downsamples[i]) if usable else 0

    def __repr__(self) -> str:
        return (
            f"SlideInfo({os.path.basename(self.path)!r}, size={self.size}, "
            f"levels={self.num_levels}, downsamples={self.level_downsamples}, mpp={self.mpp})"
        )


def open_slide(path: str | os.PathLike, backend: str = DEFAULT_BACKEND, **kwargs: Any):
    """Open a slide, returning ``(reader, wsi_handle)``.

    The handle is backend-specific and is only meaningful to the reader that
    produced it -- pass the pair around together.
    """
    from monai.data import WSIReader

    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No slide at {path}")

    reader = WSIReader(backend=backend, **kwargs)
    return reader, reader.read(path)


def get_slide_info(path: str | os.PathLike, backend: str = DEFAULT_BACKEND) -> SlideInfo:
    """Read a slide's pyramid geometry without pulling pixel data."""
    reader, wsi = open_slide(path, backend=backend)

    num_levels = reader.get_level_count(wsi)
    dims = [tuple(reader.get_size(wsi, level)[::-1]) for level in range(num_levels)]
    # get_size returns (height, width); flip so level_dimensions is (width, height)
    # to match the convention OpenSlide and QuPath both use.
    downsamples = [float(reader.get_downsample_ratio(wsi, level)) for level in range(num_levels)]

    mpp: tuple[float, float] | None = None
    power: float | None = None
    try:
        mpp_val = reader.get_mpp(wsi, 0)
        mpp = (float(mpp_val[0]), float(mpp_val[1]))
    except Exception:  # noqa: BLE001 - plenty of slides carry no resolution metadata
        logger.debug("No microns-per-pixel metadata on %s", path)
    try:
        power = float(reader.get_power(wsi, 0))
    except Exception:  # noqa: BLE001
        logger.debug("No objective power metadata on %s", path)

    return SlideInfo(os.fspath(path), dims, downsamples, mpp, power)


def read_region(
    reader: Any,
    wsi: Any,
    region: Region,
    level: int | None = None,
    channel_last: bool = True,
) -> np.ndarray:
    """Read ``region`` from an open slide.

    Args:
        reader: a ``WSIReader``, from :func:`open_slide`.
        wsi: the handle from the same call.
        region: the rectangle to read, in level-0 coordinates, with the
            ``downsample`` you want it returned at.
        level: pyramid level to read from. Defaults to the level matching
            ``region.downsample``.
        channel_last: return ``(H, W, C)``, which is what PIL, OpenCV and
            QuPath all expect. ``False`` gives MONAI's native ``(C, H, W)``.

    Returns:
        A ``uint8`` RGB array of shape ``region.mask_shape`` plus a channel axis,
        so it lines up pixel-for-pixel with masks built from the same ``Region``.
    """
    if level is None:
        level = _closest_level(reader, wsi, region.downsample)

    level_downsample = float(reader.get_downsample_ratio(wsi, int(level)))
    target_rows, target_cols = region.mask_shape

    # Size is expressed at the level being read, so derive it from the region's
    # level-0 extent and that level's downsample. Deriving it from mask_shape
    # instead would silently read the wrong *area* whenever the requested
    # downsample has no exactly matching pyramid level.
    read_rows = max(1, int(round(region.height / level_downsample)))
    read_cols = max(1, int(round(region.width / level_downsample)))

    # location is (y, x) and always in level-0 coordinates.
    data, _ = reader.get_data(
        wsi,
        location=(int(region.y), int(region.x)),
        size=(read_rows, read_cols),
        level=int(level),
    )

    array = np.asarray(data)
    if array.ndim == 3:
        array = np.moveaxis(array, 0, -1)  # (C, H, W) -> (H, W, C)

    # Close the gap between the level we could read and the resolution asked for.
    if (array.shape[0], array.shape[1]) != (target_rows, target_cols):
        array = _resize(array, target_rows, target_cols)

    if not channel_last and array.ndim == 3:
        array = np.moveaxis(array, -1, 0)
    return array


def _resize(array: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Resize an ``(H, W, C)`` uint8 array to ``(rows, cols, C)``."""
    from PIL import Image

    if array.ndim == 2:
        return np.asarray(Image.fromarray(array).resize((cols, rows), Image.BILINEAR))

    channels = [
        np.asarray(Image.fromarray(array[..., c]).resize((cols, rows), Image.BILINEAR))
        for c in range(array.shape[2])
    ]
    return np.stack(channels, axis=-1)


def _closest_level(reader: Any, wsi: Any, downsample: float) -> int:
    """Pick the pyramid level closest to ``downsample`` without going past it."""
    count = reader.get_level_count(wsi)
    ratios = [float(reader.get_downsample_ratio(wsi, i)) for i in range(count)]
    usable = [i for i, r in enumerate(ratios) if r <= downsample * 1.01]
    return max(usable, key=lambda i: ratios[i]) if usable else 0


def tissue_mask(
    thumbnail: np.ndarray,
    saturation_threshold: int = 15,
    intensity_threshold: int = 220,
) -> np.ndarray:
    """Flag tissue pixels in a low-resolution thumbnail.

    Slide backgrounds are bright and colourless while stained tissue is neither,
    so a pixel counts as tissue when it is reasonably saturated *and* not blown
    out. This is deliberately a cheap heuristic rather than Otsu: on a slide
    that is almost entirely background, Otsu picks a threshold inside the noise
    and returns a mask full of dust and scanner artefacts.

    Args:
        thumbnail: ``(H, W, 3)`` uint8 RGB, typically read at downsample 32-64.
        saturation_threshold: minimum HSV saturation (0-255) to count as stained.
        intensity_threshold: maximum RGB value; above this is glass or glare.

    Returns:
        A boolean ``(H, W)`` array, ``True`` where there is tissue.
    """
    if thumbnail.ndim != 3 or thumbnail.shape[2] < 3:
        raise ValueError(f"Expected an (H, W, 3) RGB thumbnail, got {thumbnail.shape}")

    rgb = thumbnail[..., :3].astype(np.float32)
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.where(maximum > 0, (maximum - minimum) / np.maximum(maximum, 1e-6) * 255.0, 0.0)

    return (saturation >= saturation_threshold) & (maximum < intensity_threshold)


def tile_regions(
    slide_size: tuple[int, int],
    patch_size: int | tuple[int, int] = 256,
    stride: int | tuple[int, int] | None = None,
    downsample: float = 1.0,
    bounds: Region | None = None,
) -> Iterator[Region]:
    """Generate a grid of patch regions over a slide or a sub-rectangle of one.

    Args:
        slide_size: level-0 ``(width, height)``.
        patch_size: patch size in **mask/patch** pixels, ``(w, h)`` or a scalar.
        stride: step in patch pixels. Defaults to ``patch_size`` (no overlap).
        downsample: level-0 pixels per patch pixel.
        bounds: restrict tiling to this level-0 rectangle. ``None`` tiles the
            whole slide.

    Yields:
        :class:`Region` objects in level-0 coordinates. The last row and column
        are shifted back to stay inside the slide rather than being emitted
        undersized, so every patch has identical shape -- which is what batching
        requires.
    """
    pw, ph = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    if stride is None:
        sw, sh = pw, ph
    else:
        sw, sh = (stride, stride) if isinstance(stride, int) else stride

    # Convert patch-space sizes into level-0 spans.
    span_w, span_h = int(pw * downsample), int(ph * downsample)
    step_w, step_h = int(sw * downsample), int(sh * downsample)
    if step_w <= 0 or step_h <= 0:
        raise ValueError(f"stride must be positive, got {(sw, sh)}")

    if bounds is None:
        x0, y0 = 0, 0
        x1, y1 = int(slide_size[0]), int(slide_size[1])
    else:
        x0, y0 = int(bounds.x), int(bounds.y)
        x1, y1 = x0 + int(bounds.width), y0 + int(bounds.height)
        x1, y1 = min(x1, int(slide_size[0])), min(y1, int(slide_size[1]))

    if x1 - x0 < span_w or y1 - y0 < span_h:
        logger.warning(
            "Tiling area %dx%d is smaller than one %dx%d patch; no regions produced.",
            x1 - x0, y1 - y0, span_w, span_h,
        )
        return

    for y in _axis_origins(y0, y1, span_h, step_h):
        for x in _axis_origins(x0, x1, span_w, step_w):
            yield Region(x=x, y=y, width=span_w, height=span_h, downsample=downsample)


def _axis_origins(start: int, end: int, span: int, step: int) -> list[int]:
    """Tile origins along one axis, covering ``[start, end)`` completely.

    A plain ``range(start, end - span + 1, step)`` stops short whenever the axis
    length is not an exact multiple of the step, leaving up to ``span - 1``
    pixels at the far edge untiled -- nuclei there would never be trained on or
    inferred. The final origin is therefore pulled back to ``end - span`` so the
    last tile ends exactly on the edge. That tile overlaps its neighbour, which
    is fine for inference (predictions are stitched) and for training (patches
    are independent samples), and it keeps every tile the same shape, which is
    what batching needs.
    """
    if end - start < span:
        return []
    origins = list(range(start, end - span + 1, step))
    if origins[-1] != end - span:
        origins.append(end - span)
    return origins
