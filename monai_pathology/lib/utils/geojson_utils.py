"""Conversion between QuPath GeoJSON annotations and raster masks.

This is the contract between QuPath and MONAI, and it runs in both directions:

- **Inbound** (training): polygons the user drew in QuPath become label masks for
  patches cut out of the slide.
- **Outbound** (inference): predicted masks become polygons QuPath can display.

Both directions live here on purpose. The offset/scale arithmetic between
whole-slide coordinates and patch-local pixels is the single easiest thing in
this project to get subtly wrong -- an off-by-one or a forgotten downsample
produces annotations that look plausible but sit a few pixels off, which is
miserable to debug from either end. One module, one set of tests.

Coordinate conventions, which differ between the systems involved:

- QuPath GeoJSON stores ``(x, y)`` pairs in **level-0** (full resolution) slide
  pixels, regardless of what magnification you were viewing.
- NumPy masks index as ``[row, col]``, i.e. ``[y, x]``.
- MONAI's :class:`~monai.data.WSIReader` takes ``location=(row, col)``, i.e.
  ``(y, x)`` -- the opposite order from GeoJSON. See :mod:`wsi_utils`.

:class:`Region` carries the offset and downsample that relate the first two, so
callers never do the arithmetic by hand.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "Region",
    "Annotation",
    "load_geojson",
    "save_geojson",
    "parse_annotations",
    "annotations_to_mask",
    "mask_to_annotations",
    "annotations_to_geojson",
]


@dataclass(frozen=True)
class Region:
    """A rectangular slide region and the resolution a mask of it is stored at.

    Args:
        x: level-0 x (column) of the region's top-left corner.
        y: level-0 y (row) of the region's top-left corner.
        width: region width in level-0 pixels.
        height: region height in level-0 pixels.
        downsample: level-0 pixels per mask pixel. 1.0 means the mask is at full
            resolution; 4.0 means the mask is 4x smaller in each dimension, as
            you would get reading at a higher pyramid level.
    """

    x: int
    y: int
    width: int
    height: int
    downsample: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Region must have positive size, got {self.width}x{self.height}")
        if self.downsample <= 0:
            raise ValueError(f"downsample must be positive, got {self.downsample}")

    @property
    def mask_shape(self) -> tuple[int, int]:
        """Mask shape as ``(rows, cols)``."""
        return (
            int(round(self.height / self.downsample)),
            int(round(self.width / self.downsample)),
        )

    def to_local(self, coords: np.ndarray) -> np.ndarray:
        """Map level-0 ``(x, y)`` pairs into mask-local ``(col, row)`` pairs."""
        coords = np.asarray(coords, dtype=np.float64)
        return (coords - np.array([self.x, self.y], dtype=np.float64)) / self.downsample

    def to_slide(self, coords: np.ndarray) -> np.ndarray:
        """Map mask-local ``(col, row)`` pairs back to level-0 ``(x, y)`` pairs."""
        coords = np.asarray(coords, dtype=np.float64)
        return coords * self.downsample + np.array([self.x, self.y], dtype=np.float64)


@dataclass
class Annotation:
    """One annotated object: an outline plus what it was called.

    Args:
        exterior: ``(N, 2)`` array of level-0 ``(x, y)`` vertices.
        interiors: holes, each an ``(M, 2)`` array in the same coordinate space.
        label: class name, e.g. ``"Nucleus"``. ``None`` means unclassified --
            QuPath lets you draw without assigning a class.
        properties: the feature's original GeoJSON properties, kept so that
            round-tripping through this module does not silently drop metadata.
    """

    exterior: np.ndarray
    interiors: list[np.ndarray] = field(default_factory=list)
    label: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# File I/O
# --------------------------------------------------------------------------


def load_geojson(path: str | os.PathLike) -> dict[str, Any]:
    """Read a GeoJSON file, tolerating the shapes QuPath actually emits.

    QuPath exports either a ``FeatureCollection`` or, for some export paths, a
    bare JSON list of features. Both are normalised to a ``FeatureCollection``.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return {"type": "FeatureCollection", "features": data}
    if isinstance(data, dict) and data.get("type") == "Feature":
        return {"type": "FeatureCollection", "features": [data]}
    if isinstance(data, dict) and "features" in data:
        return data

    raise ValueError(
        f"{path} is not GeoJSON this module recognises "
        "(expected a FeatureCollection, a Feature, or a list of Features)."
    )


def save_geojson(geojson: dict[str, Any], path: str | os.PathLike, indent: int | None = None) -> None:
    """Write a FeatureCollection.

    ``indent=None`` by default: a slide can carry hundreds of thousands of
    nuclei, and pretty-printing that is a real cost in both size and time.
    """
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(geojson, handle, indent=indent)


# --------------------------------------------------------------------------
# GeoJSON -> annotations
# --------------------------------------------------------------------------


def _classification_name(properties: dict[str, Any]) -> str | None:
    """Pull the class name out of a QuPath feature's properties.

    QuPath has moved this around across versions: modern builds nest it under
    ``classification.name``, older ones used a flat ``name``, and the newer
    object model sometimes carries it as ``classification`` as a bare string.
    """
    classification = properties.get("classification")
    if isinstance(classification, dict):
        name = classification.get("name")
        if name:
            return str(name)
    elif isinstance(classification, str) and classification:
        return classification

    for key in ("name", "class", "label"):
        value = properties.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _rings_from_geometry(geometry: dict[str, Any]) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Flatten a GeoJSON geometry into ``(exterior, interiors)`` pairs.

    Handles Polygon, MultiPolygon, and GeometryCollection. Point/LineString
    geometries are skipped -- QuPath can hold them, but they carry no area and
    so cannot become mask pixels.
    """
    gtype = geometry.get("type")

    if gtype == "Polygon":
        rings = geometry.get("coordinates") or []
        if not rings:
            return []
        exterior = np.asarray(rings[0], dtype=np.float64)
        interiors = [np.asarray(r, dtype=np.float64) for r in rings[1:]]
        return [(exterior, interiors)]

    if gtype == "MultiPolygon":
        out: list[tuple[np.ndarray, list[np.ndarray]]] = []
        for poly in geometry.get("coordinates") or []:
            if not poly:
                continue
            exterior = np.asarray(poly[0], dtype=np.float64)
            interiors = [np.asarray(r, dtype=np.float64) for r in poly[1:]]
            out.append((exterior, interiors))
        return out

    if gtype == "GeometryCollection":
        out = []
        for sub in geometry.get("geometries") or []:
            out.extend(_rings_from_geometry(sub))
        return out

    if gtype in ("Point", "MultiPoint", "LineString", "MultiLineString"):
        logger.debug("Skipping zero-area geometry of type %s", gtype)
        return []

    logger.warning("Ignoring unsupported geometry type %r", gtype)
    return []


def parse_annotations(
    geojson: dict[str, Any],
    keep_labels: Iterable[str] | None = None,
) -> list[Annotation]:
    """Turn a FeatureCollection into :class:`Annotation` objects.

    Args:
        geojson: as returned by :func:`load_geojson`.
        keep_labels: if given, only annotations whose label is in this set are
            returned. Useful when a slide carries several annotation classes
            but the model being trained only cares about one.
    """
    keep = set(keep_labels) if keep_labels is not None else None
    annotations: list[Annotation] = []

    for feature in geojson.get("features") or []:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        properties = feature.get("properties") or {}
        label = _classification_name(properties)
        if keep is not None and label not in keep:
            continue

        for exterior, interiors in _rings_from_geometry(geometry):
            if len(exterior) < 3:
                continue
            annotations.append(
                Annotation(
                    exterior=exterior,
                    interiors=interiors,
                    label=label,
                    properties=properties,
                )
            )

    return annotations


# --------------------------------------------------------------------------
# Annotations -> mask
# --------------------------------------------------------------------------


def annotations_to_mask(
    annotations: Sequence[Annotation],
    region: Region,
    label_map: dict[str, int] | None = None,
    default_value: int = 1,
    dtype: np.dtype | type = np.uint8,
) -> np.ndarray:
    """Rasterise annotations into a mask covering ``region``.

    Args:
        annotations: level-0 polygons, e.g. from :func:`parse_annotations`.
        region: the slide rectangle and resolution to rasterise at.
        label_map: class name -> pixel value. Annotations whose label is absent
            from the map are skipped; pass ``None`` to burn every annotation in
            with ``default_value`` (the binary case).
        default_value: value used for unclassified annotations, or for all of
            them when ``label_map`` is ``None``.
        dtype: mask dtype. ``uint8`` suits up to 255 classes.

    Returns:
        A ``(rows, cols)`` array. Later annotations overwrite earlier ones where
        they overlap, so pass them in ascending priority order if that matters.

    Note:
        Polygons are clipped to the region implicitly -- anything falling
        outside simply lands off the canvas.
    """
    from PIL import Image, ImageDraw  # local import: keeps module import cheap

    rows, cols = region.mask_shape
    canvas = Image.new("I", (cols, rows), 0)  # 32-bit so class values never clip
    draw = ImageDraw.Draw(canvas)

    drawn = 0
    for ann in annotations:
        if label_map is not None:
            if ann.label is None or ann.label not in label_map:
                continue
            value = int(label_map[ann.label])
        else:
            value = int(default_value)

        exterior = region.to_local(ann.exterior)
        if len(exterior) < 3:
            continue
        draw.polygon([(float(px), float(py)) for px, py in exterior], fill=value, outline=value)

        # Punch out holes after the exterior, or they would be filled over.
        for interior in ann.interiors:
            ring = region.to_local(interior)
            if len(ring) >= 3:
                draw.polygon([(float(px), float(py)) for px, py in ring], fill=0, outline=0)
        drawn += 1

    logger.debug("Rasterised %d/%d annotations into %s", drawn, len(annotations), region.mask_shape)
    return np.asarray(canvas, dtype=np.int32).astype(dtype)


# --------------------------------------------------------------------------
# Mask -> annotations
# --------------------------------------------------------------------------


def _iter_instances(class_mask: np.ndarray) -> Iterator[tuple[np.ndarray, tuple[int, int]]]:
    """Yield ``(sub_mask, (x_offset, y_offset))`` per connected component.

    Each component is cropped to its own bounding box rather than returned as a
    full-tile boolean array. A dense tile can hold thousands of nuclei, and one
    tile-sized mask each puts peak memory at ``objects x tile area`` -- roughly
    131 MB for a 512x512 tile with 500 nuclei, repeated for every tile in the
    slide. Cropping makes it proportional to the largest single object instead.

    The crop is padded by one pixel so that a component touching its bounding
    box edge traces the same contour it would have in the full tile; the offset
    accounts for that padding.
    """
    from scipy import ndimage

    # 3x3 structure == skimage's connectivity=2 (diagonals count as connected).
    components, count = ndimage.label(class_mask, structure=np.ones((3, 3), dtype=bool))
    if not count:
        return

    for index, bbox in enumerate(ndimage.find_objects(components), start=1):
        if bbox is None:
            continue
        rows, cols = bbox
        sub = np.zeros((rows.stop - rows.start + 2, cols.stop - cols.start + 2), dtype=bool)
        sub[1:-1, 1:-1] = components[rows, cols] == index
        yield sub, (cols.start - 1, rows.start - 1)


def _find_contours(binary: np.ndarray) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Trace a binary mask into ``(exterior, interiors)`` pairs of ``(col, row)``.

    Uses OpenCV's ``RETR_CCOMP``, which returns a two-level hierarchy: outer
    boundaries and the holes inside them. Keeping holes matters for tissue
    regions, where a lumen or a tear inside an annotated area is real and should
    not be filled in.

    Falls back to scikit-image when OpenCV is unavailable. The fallback has no
    hierarchy information, so it fills holes rather than risk the alternative:
    ``skimage.measure.find_contours`` traces a hole's boundary identically to an
    object's, which would turn every hole into a spurious positive object.
    Losing a hole is a much smaller error than inventing a nucleus.
    """
    binary = binary.astype(np.uint8)
    try:
        import cv2
    except ImportError:
        from scipy import ndimage
        from skimage import measure

        filled = ndimage.binary_fill_holes(binary)
        out: list[tuple[np.ndarray, list[np.ndarray]]] = []
        # find_contours returns (row, col); flip to (col, row) for GeoJSON.
        for contour in measure.find_contours(filled.astype(np.float32), level=0.5):
            if len(contour) >= 3:
                out.append((contour[:, ::-1].astype(np.float64), []))
        return out

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return []

    # hierarchy rows are [next, previous, first_child, parent]; under RETR_CCOMP
    # a parent of -1 means an outer boundary, anything else means a hole.
    hierarchy = hierarchy[0]
    results: list[tuple[np.ndarray, list[np.ndarray]]] = []
    index_of: dict[int, int] = {}

    for i, (contour, node) in enumerate(zip(contours, hierarchy)):
        if node[3] != -1:
            continue
        points = contour.reshape(-1, 2).astype(np.float64)  # already (x, y)
        if len(points) < 3:
            continue
        index_of[i] = len(results)
        results.append((points, []))

    for i, (contour, node) in enumerate(zip(contours, hierarchy)):
        parent = int(node[3])
        if parent == -1 or parent not in index_of:
            continue
        points = contour.reshape(-1, 2).astype(np.float64)
        if len(points) >= 3:
            results[index_of[parent]][1].append(points)

    return results


def mask_to_annotations(
    mask: np.ndarray,
    region: Region,
    label_map: dict[str, int] | None = None,
    separate_instances: bool = True,
    min_area: float = 0.0,
    simplify_tolerance: float = 0.0,
) -> list[Annotation]:
    """Vectorise a mask back into level-0 polygons.

    Args:
        mask: ``(rows, cols)`` integer mask. Zero is background.
        region: the region this mask covers, used to map back to slide space.
        label_map: class name -> pixel value, the same mapping used on the way
            in. Inverted here to name the outputs. ``None`` labels everything
            positive as ``"Region"``.
        separate_instances: split each connected component into its own
            annotation. This is what turns a semantic nucleus mask into
            countable objects, which is usually the point for nuclei. Set
            ``False`` for tissue regions, where one big polygon per class is
            more useful than thousands of fragments.
        min_area: drop objects smaller than this, in **mask** pixels. The single
            most effective noise filter on raw model output.
        simplify_tolerance: Douglas-Peucker tolerance in mask pixels. Contours
            traced from a raster are stair-stepped and vertex-heavy; a tolerance
            around 1.0 typically cuts GeoJSON size several-fold with no visible
            change in QuPath. 0 disables simplification.

    Returns:
        Annotations in level-0 slide coordinates, ready for
        :func:`annotations_to_geojson`.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]  # tolerate a leading channel dim from MONAI
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")

    value_to_label = {int(v): k for k, v in (label_map or {}).items()}
    annotations: list[Annotation] = []

    for value in np.unique(mask):
        value = int(value)
        if value == 0:
            continue
        label = value_to_label.get(value, "Region" if label_map is None else None)
        class_mask = mask == value

        pieces = _iter_instances(class_mask) if separate_instances else iter([(class_mask, (0, 0))])

        for piece, offset in pieces:
            if min_area > 0 and int(piece.sum()) < min_area:
                continue
            for exterior, interiors in _find_contours(piece):
                exterior = _maybe_simplify(exterior + offset, simplify_tolerance)
                if len(exterior) < 3:
                    continue
                rings = []
                for interior in interiors:
                    interior = _maybe_simplify(interior + offset, simplify_tolerance)
                    if len(interior) >= 3:
                        rings.append(region.to_slide(interior))
                annotations.append(
                    Annotation(
                        exterior=region.to_slide(exterior),
                        interiors=rings,
                        label=label,
                        properties={"objectType": "annotation"},
                    )
                )

    return annotations


def _maybe_simplify(contour: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker simplification, if shapely is available and asked for."""
    if tolerance <= 0:
        return contour
    try:
        from shapely.geometry import Polygon

        simplified = Polygon(contour).simplify(tolerance, preserve_topology=True)
        if simplified.is_empty or simplified.geom_type != "Polygon":
            return contour
        return np.asarray(simplified.exterior.coords, dtype=np.float64)
    except Exception:  # noqa: BLE001 - simplification is optional, never fatal
        logger.debug("Simplification failed; keeping the full contour", exc_info=True)
        return contour


# --------------------------------------------------------------------------
# Annotations -> GeoJSON
# --------------------------------------------------------------------------


def annotations_to_geojson(
    annotations: Sequence[Annotation],
    colors: dict[str, Sequence[int]] | None = None,
    object_type: str = "annotation",
) -> dict[str, Any]:
    """Build a FeatureCollection QuPath will import.

    Args:
        annotations: level-0 polygons.
        colors: class name -> ``[r, g, b]``. QuPath assigns its own colours for
            classes it does not know, so this is cosmetic.
        object_type: QuPath's ``objectType`` property. ``"annotation"`` makes
            editable objects; ``"detection"`` makes lightweight ones better
            suited to the tens of thousands of nuclei a slide-wide run produces.

    Returns:
        A FeatureCollection dict, ready for :func:`save_geojson`.
    """
    colors = colors or {}
    features: list[dict[str, Any]] = []

    for ann in annotations:
        exterior = np.asarray(ann.exterior, dtype=np.float64)
        if len(exterior) < 3:
            continue

        # GeoJSON requires linear rings to be explicitly closed.
        rings = [_closed_ring(exterior)]
        rings.extend(_closed_ring(np.asarray(r, dtype=np.float64)) for r in ann.interiors)

        properties: dict[str, Any] = {"objectType": object_type}
        if ann.label:
            classification: dict[str, Any] = {"name": ann.label}
            if ann.label in colors:
                classification["color"] = list(colors[ann.label])
            properties["classification"] = classification

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": rings},
                "properties": properties,
            }
        )

    return {"type": "FeatureCollection", "features": features}


def _closed_ring(coords: np.ndarray) -> list[list[float]]:
    """Return ``coords`` as a closed ring of ``[x, y]`` lists."""
    if len(coords) and not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[:1]])
    return [[float(x), float(y)] for x, y in coords]
