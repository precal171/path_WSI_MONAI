"""WSI reading, tiling and tissue detection against a synthetic pyramidal slide.

The coordinate-order test is the important one here. MONAI's ``WSIReader`` takes
``location=(row, col)`` while GeoJSON is ``(x, y)``; if that flip is ever wrong,
patches and their masks come from different parts of the slide and the model
trains on mismatched pairs -- which shows up as a model that simply never learns,
with nothing obviously broken anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from monai_pathology.lib.utils.geojson_utils import Region
from monai_pathology.lib.utils.wsi_utils import (
    get_slide_info,
    open_slide,
    read_region,
    tile_regions,
    tissue_mask,
)

BACKEND = "TiffFile"  # OpenSlide only recognises pyramids in vendor formats


class TestSlideInfo:
    def test_reports_geometry(self, synthetic_slide):
        slide_path, _ = synthetic_slide
        info = get_slide_info(slide_path, backend=BACKEND)
        assert info.size == (1024, 1024)
        assert info.num_levels >= 1
        assert info.level_downsamples[0] == 1.0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            get_slide_info("/nonexistent/slide.svs", backend=BACKEND)


class TestReadRegion:
    def test_coordinate_order_matches_numpy_indexing(self, synthetic_slide):
        """read_region(Region(x, y, ...)) must equal full[y:y+h, x:x+w]."""
        slide_path, _ = synthetic_slide
        reader, wsi = open_slide(slide_path, backend=BACKEND)

        full, _ = reader.get_data(wsi, location=(0, 0), size=(1024, 1024), level=0)
        full = np.moveaxis(np.asarray(full), 0, -1)

        patch = read_region(reader, wsi, Region(x=256, y=128, width=128, height=64))
        np.testing.assert_array_equal(patch, full[128:192, 256:384])

    def test_shape_and_dtype(self, synthetic_slide):
        slide_path, _ = synthetic_slide
        reader, wsi = open_slide(slide_path, backend=BACKEND)
        patch = read_region(reader, wsi, Region(0, 0, 256, 256))
        assert patch.shape == (256, 256, 3)
        assert patch.dtype == np.uint8

    def test_channel_first_option(self, synthetic_slide):
        slide_path, _ = synthetic_slide
        reader, wsi = open_slide(slide_path, backend=BACKEND)
        assert read_region(reader, wsi, Region(0, 0, 64, 64), channel_last=False).shape == (3, 64, 64)

    @pytest.mark.parametrize("downsample,expected", [(1.0, 256), (2.0, 128), (4.0, 64)])
    def test_downsample_changes_resolution_not_area(self, synthetic_slide, downsample, expected):
        """A downsampled read must cover the same slide area, just smaller.

        Regression test: deriving the read size from the mask shape rather than
        the region's level-0 extent silently returned a full-resolution crop of
        the wrong area whenever no pyramid level matched exactly.
        """
        slide_path, _ = synthetic_slide
        reader, wsi = open_slide(slide_path, backend=BACKEND)

        patch = read_region(reader, wsi, Region(0, 0, 256, 256, downsample))
        assert patch.shape == (expected, expected, 3)

        full = read_region(reader, wsi, Region(0, 0, 256, 256, 1.0))
        # Same area at a different scale => same average colour.
        np.testing.assert_allclose(
            patch.astype(float).mean(axis=(0, 1)), full.astype(float).mean(axis=(0, 1)), atol=4.0
        )

    def test_non_power_of_two_downsample(self, synthetic_slide):
        slide_path, _ = synthetic_slide
        reader, wsi = open_slide(slide_path, backend=BACKEND)
        assert read_region(reader, wsi, Region(0, 0, 300, 300, 3.0)).shape == (100, 100, 3)


class TestTiling:
    def test_covers_the_slide_without_overlap(self):
        tiles = list(tile_regions((1024, 1024), patch_size=256))
        assert len(tiles) == 16
        assert tiles[0] == Region(0, 0, 256, 256, 1.0)
        assert all(t.x + t.width <= 1024 and t.y + t.height <= 1024 for t in tiles)

    def test_stride_creates_overlap(self):
        assert len(list(tile_regions((1024, 1024), 256, stride=128))) > 16

    def test_downsample_widens_each_tile(self):
        tiles = list(tile_regions((1024, 1024), patch_size=256, downsample=2.0))
        assert tiles[0].width == 512      # 256 patch px * 2
        assert tiles[0].mask_shape == (256, 256)

    def test_bounds_restrict_the_area(self):
        bounds = Region(256, 256, 512, 512, 1.0)
        tiles = list(tile_regions((1024, 1024), 256, bounds=bounds))
        assert len(tiles) == 4
        assert all(256 <= t.x < 768 and 256 <= t.y < 768 for t in tiles)

    def test_area_smaller_than_a_patch_yields_nothing(self):
        assert list(tile_regions((100, 100), patch_size=256)) == []


class TestTissueMask:
    def test_flags_stained_pixels_only(self):
        thumbnail = np.full((50, 50, 3), 245, dtype=np.uint8)   # bright glass
        thumbnail[10:30, 10:30] = [150, 90, 160]                 # stained tissue

        mask = tissue_mask(thumbnail)
        assert mask[20, 20]
        assert not mask[45, 45]
        assert 0.1 < mask.mean() < 0.3

    def test_finds_tissue_in_the_synthetic_slide(self, synthetic_slide):
        slide_path, _ = synthetic_slide
        reader, wsi = open_slide(slide_path, backend=BACKEND)
        thumbnail = read_region(reader, wsi, Region(0, 0, 1024, 1024, 8.0))
        assert 0.0 < tissue_mask(thumbnail).mean() < 1.0

    def test_rejects_non_rgb_input(self):
        with pytest.raises(ValueError):
            tissue_mask(np.zeros((10, 10), dtype=np.uint8))
