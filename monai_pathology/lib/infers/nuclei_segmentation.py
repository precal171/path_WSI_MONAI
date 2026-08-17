"""Interactive nucleus segmentation for the MONAI Label server.

QuPath asks the server to run the model over the region currently on screen; the
server returns GeoJSON, which QuPath draws as annotation objects.

The structure here is deliberate. All of the real work lives in
:meth:`NucleiSegmentationInfer.infer_region`, which takes a slide path and a
:class:`Region` and returns GeoJSON -- no MONAI Label types in its signature, so
it can be tested in isolation without a server, a GPU or the ``monailabel``
package installed. :meth:`__call__` is the thin adapter that unpacks MONAI
Label's request dict and calls it.

That split matters because MONAI Label's request/response contract is the part
of this project most likely to shift between releases. When it does, the fix is
confined to ``__call__``, and the segmentation logic underneath is unaffected.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

from .._compat import BasicInferTask, InferType
from ..utils.geojson_utils import Region, annotations_to_geojson, mask_to_annotations, save_geojson
from ..utils.wsi_utils import open_slide, read_region

logger = logging.getLogger(__name__)


class NucleiSegmentationInfer(BasicInferTask):
    """Runs the nuclei bundle over a slide region and returns QuPath GeoJSON."""

    def __init__(
        self,
        path: str,
        bundle_root: str,
        labels: dict[str, int] | None = None,
        roi_size: tuple[int, int] = (256, 256),
        downsample: float = 1.0,
        backend: str = "OpenSlide",
        min_poly_area: float = 10.0,
        simplify_tolerance: float = 1.0,
        dimension: int = 2,
        description: str = "Nucleus segmentation on whole-slide images",
        **kwargs: Any,
    ) -> None:
        self.bundle_root = bundle_root
        self.backend = backend
        self.downsample = downsample
        self.min_poly_area = min_poly_area
        self.simplify_tolerance = simplify_tolerance
        self._pieces: dict[str, Any] | None = None

        super().__init__(
            path=path,
            network=None,
            type=InferType.SEGMENTATION,
            labels=labels or {"Nucleus": 1},
            dimension=dimension,
            description=description,
            **kwargs,
        )
        self.roi_size = roi_size

    # ------------------------------------------------------------------
    # Bundle wiring
    # ------------------------------------------------------------------

    def _load_pieces(self, device: torch.device) -> dict[str, Any]:
        """Resolve network + transforms from the bundle, once per process.

        Cached because MONAI Label calls inference once per user interaction and
        rebuilding the network each time would make the round trip feel slow in
        QuPath.
        """
        if self._pieces is not None:
            return self._pieces

        import sys

        from monai.bundle import ConfigParser

        if self.bundle_root not in sys.path:
            sys.path.insert(0, self.bundle_root)

        parser = ConfigParser()
        parser.read_config(os.path.join(self.bundle_root, "configs", "inference.json"))
        parser.update({"bundle_root": self.bundle_root, "device": device})

        network = parser.get_parsed_content("network")

        from scripts.utils import load_weights  # bundle-local helper

        load_weights(network, self.path if isinstance(self.path, str) else self.path[0], device)
        network.eval()

        self._pieces = {
            "network": network,
            "preprocessing": parser.get_parsed_content("array_preprocessing"),
            "inferer": parser.get_parsed_content("inferer"),
            "threshold": float(parser.get_parsed_content("threshold")),
        }
        return self._pieces

    # ------------------------------------------------------------------
    # The actual work -- no MONAI Label types involved
    # ------------------------------------------------------------------

    def infer_region(
        self,
        image_path: str,
        region: Region,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        """Segment nuclei in one slide region.

        Args:
            image_path: path to the slide.
            region: level-0 rectangle to run over, with the downsample to run at.
            device: where to run. Defaults to CUDA when available.

        Returns:
            A GeoJSON FeatureCollection in level-0 slide coordinates.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pieces = self._load_pieces(device)

        reader, wsi = open_slide(image_path, backend=self.backend)
        patch = read_region(reader, wsi, region)
        if patch.ndim == 3 and patch.shape[2] > 3:
            patch = patch[..., :3]

        data = pieces["preprocessing"]({"image": np.moveaxis(patch, -1, 0).astype(np.float32)})
        tensor = data["image"] if isinstance(data, dict) else data
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(np.asarray(tensor))

        batch = tensor.float().unsqueeze(0).to(device)

        # QuPath decides how big a region to ask for, and it can be smaller than
        # the sliding window. When it is, MONAI pads the input out to roi_size
        # and the network sees a border of constant padding it never met during
        # training; predictions come back under-confident enough to vanish at the
        # threshold, with nothing raised. Shrink the window to fit instead.
        inferer = pieces["inferer"]
        height, width = int(batch.shape[-2]), int(batch.shape[-1])
        roi = getattr(inferer, "roi_size", None)
        if roi is not None and (int(roi[0]) > height or int(roi[1]) > width):
            from monai.inferers import SlidingWindowInferer

            logger.debug("Shrinking inference window from %s to (%d, %d)", list(roi), height, width)
            inferer = SlidingWindowInferer(
                roi_size=(min(int(roi[0]), height), min(int(roi[1]), width)),
                sw_batch_size=getattr(inferer, "sw_batch_size", 4),
                overlap=getattr(inferer, "overlap", 0.25),
            )

        with torch.no_grad():
            logits = inferer(batch, pieces["network"])
            probabilities = torch.sigmoid(logits)

        mask = (probabilities[0, 0] >= pieces["threshold"]).to(torch.uint8).cpu().numpy()

        label_name = next(iter(self.labels)) if self.labels else "Nucleus"
        annotations = mask_to_annotations(
            mask,
            region,
            label_map={label_name: 1},
            separate_instances=True,
            min_area=self.min_poly_area,
            simplify_tolerance=self.simplify_tolerance,
        )
        logger.info("Segmented %d objects in %s", len(annotations), region)

        return annotations_to_geojson(
            annotations, colors={label_name: [200, 0, 200]}, object_type="annotation"
        )

    # ------------------------------------------------------------------
    # MONAI Label adapter
    # ------------------------------------------------------------------

    @staticmethod
    def _region_from_request(request: dict[str, Any], downsample: float) -> Region:
        """Build a :class:`Region` from a MONAI Label / QuPath inference request.

        QuPath's extension sends ``location`` as ``(x, y)`` and ``size`` as
        ``(width, height)`` in level-0 coordinates. Both are normalised here.
        """
        location = request.get("location") or [0, 0]
        size = request.get("size") or [1024, 1024]
        level = int(request.get("level", 0) or 0)

        # A non-zero level means QuPath is asking at reduced resolution.
        effective_downsample = float(request.get("downsample", downsample) or downsample)
        if level > 0:
            effective_downsample = max(effective_downsample, float(2**level))

        return Region(
            x=int(location[0]),
            y=int(location[1]),
            width=int(size[0]),
            height=int(size[1]),
            downsample=effective_downsample,
        )

    def __call__(self, request: dict[str, Any], **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Handle an inference request from the server.

        Returns ``(result_file, result_json)``, which is MONAI Label's contract:
        the file is written GeoJSON, and the JSON payload carries the same
        annotations inline for clients that prefer them in the response body.

        Note:
            The exact keys MONAI Label and the QuPath extension exchange have
            varied across releases. If QuPath shows no annotations, log
            ``request`` here first -- the geometry is usually right and the
            wrapping is what differs.
        """
        image_path = request.get("image")
        if isinstance(image_path, (list, tuple)):
            image_path = image_path[0]
        if not image_path or not os.path.exists(str(image_path)):
            raise ValueError(f"Inference request did not include a readable slide path: {image_path!r}")

        region = self._region_from_request(request, self.downsample)
        device_name = request.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        geojson = self.infer_region(str(image_path), region, torch.device(device_name))

        output_file = request.get("result_file_name")
        if not output_file:
            import tempfile

            handle = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False)
            handle.close()
            output_file = handle.name
        save_geojson(geojson, output_file)

        return output_file, {"annotation": geojson, "label_names": self.labels}

    # ------------------------------------------------------------------
    # BasicInferTask abstract members
    # ------------------------------------------------------------------
    # Inference is driven through the bundle in __call__ rather than through the
    # base class's transform pipeline, so these exist to satisfy the interface.

    def pre_transforms(self, data: Any = None) -> list[Any]:
        return []

    def post_transforms(self, data: Any = None) -> list[Any]:
        return []

    def inferer(self, data: Any = None) -> Any:
        from monai.inferers import SlidingWindowInferer

        return SlidingWindowInferer(roi_size=self.roi_size, sw_batch_size=4, overlap=0.25)

    def is_valid(self) -> bool:
        """Report whether trained weights exist yet.

        MONAI Label uses this to decide whether to advertise the model, which
        keeps QuPath from offering inference that would only return noise.
        """
        path = self.path[0] if isinstance(self.path, (list, tuple)) else self.path
        return bool(path) and os.path.exists(str(path))
