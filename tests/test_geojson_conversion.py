"""GeoJSON <-> mask conversion.

The offset/scale arithmetic between slide and patch coordinates is the easiest
thing in this project to get quietly wrong, so these tests check actual
coordinates rather than just that something came back.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from monai_pathology.lib.utils.geojson_utils import (
    Annotation,
    Region,
    annotations_to_geojson,
    annotations_to_mask,
    load_geojson,
    mask_to_annotations,
    parse_annotations,
    save_geojson,
)


def square(x0: float, y0: float, size: float) -> np.ndarray:
    return np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]], dtype=float)


class TestRegion:
    def test_mask_shape_accounts_for_downsample(self):
        assert Region(0, 0, 400, 200, 1.0).mask_shape == (200, 400)
        assert Region(0, 0, 400, 200, 2.0).mask_shape == (100, 200)

    def test_coordinate_round_trip(self):
        region = Region(x=1000, y=2000, width=400, height=400, downsample=2.0)
        slide_coords = np.array([[1000.0, 2000.0], [1200.0, 2100.0]])
        local = region.to_local(slide_coords)
        np.testing.assert_allclose(local, [[0.0, 0.0], [100.0, 50.0]])
        np.testing.assert_allclose(region.to_slide(local), slide_coords)

    def test_rejects_nonsense_geometry(self):
        with pytest.raises(ValueError):
            Region(0, 0, 0, 100)
        with pytest.raises(ValueError):
            Region(0, 0, 100, 100, downsample=0)


class TestRasterisation:
    def test_polygon_lands_in_the_right_place(self):
        region = Region(0, 0, 100, 100, 1.0)
        mask = annotations_to_mask([Annotation(exterior=square(20, 30, 40), label="N")], region, {"N": 1})

        assert mask.shape == (100, 100)
        assert mask[50, 40] == 1  # inside
        assert mask[10, 10] == 0  # outside
        # Rows 30..70, cols 20..60 -- i.e. [y, x], not [x, y].
        rows, cols = np.nonzero(mask)
        assert 29 <= rows.min() <= 31 and 69 <= rows.max() <= 71
        assert 19 <= cols.min() <= 21 and 59 <= cols.max() <= 61

    def test_offset_region_shifts_coordinates(self):
        region = Region(x=1000, y=2000, width=100, height=100, downsample=1.0)
        mask = annotations_to_mask([Annotation(exterior=square(1020, 2030, 40), label="N")], region, {"N": 1})
        rows, cols = np.nonzero(mask)
        assert 29 <= rows.min() <= 31
        assert 19 <= cols.min() <= 21

    def test_downsample_scales_coordinates(self):
        region = Region(0, 0, 200, 200, 2.0)
        mask = annotations_to_mask([Annotation(exterior=square(40, 60, 80), label="N")], region, {"N": 1})
        assert mask.shape == (100, 100)
        rows, cols = np.nonzero(mask)
        assert 29 <= rows.min() <= 31   # 60 / 2
        assert 19 <= cols.min() <= 21   # 40 / 2

    def test_holes_are_punched_out(self):
        region = Region(0, 0, 100, 100, 1.0)
        annotation = Annotation(exterior=square(10, 10, 80), interiors=[square(40, 40, 20)], label="N")
        mask = annotations_to_mask([annotation], region, {"N": 1})
        assert mask[50, 50] == 0   # inside the hole
        assert mask[20, 20] == 1   # inside the ring

    def test_multiclass_values(self):
        region = Region(0, 0, 100, 100, 1.0)
        annotations = [
            Annotation(exterior=square(5, 5, 20), label="Tumor"),
            Annotation(exterior=square(50, 50, 20), label="Stroma"),
        ]
        mask = annotations_to_mask(annotations, region, {"Tumor": 1, "Stroma": 2})
        assert set(np.unique(mask)) == {0, 1, 2}

    def test_unmapped_labels_are_skipped(self):
        region = Region(0, 0, 100, 100, 1.0)
        annotations = [
            Annotation(exterior=square(5, 5, 20), label="Tumor"),
            Annotation(exterior=square(50, 50, 20), label="Artifact"),
        ]
        mask = annotations_to_mask(annotations, region, {"Tumor": 1})
        assert set(np.unique(mask)) == {0, 1}
        assert mask[60, 60] == 0


class TestVectorisation:
    def test_round_trip_preserves_position_and_area(self):
        region = Region(x=500, y=700, width=200, height=200, downsample=1.0)
        original = Annotation(exterior=square(560, 760, 60), label="Nucleus")

        mask = annotations_to_mask([original], region, {"Nucleus": 1})
        recovered = mask_to_annotations(mask, region, {"Nucleus": 1}, min_area=5)

        assert len(recovered) == 1
        assert recovered[0].label == "Nucleus"

        # Within a pixel of the original, in absolute slide coordinates.
        coords = recovered[0].exterior
        assert abs(coords[:, 0].min() - 560) <= 1.5
        assert abs(coords[:, 1].min() - 760) <= 1.5
        assert abs(coords[:, 0].max() - 620) <= 1.5

        # Re-rasterising should reproduce the same mask almost exactly.
        remask = annotations_to_mask(recovered, region, {"Nucleus": 1})
        iou = np.logical_and(mask > 0, remask > 0).sum() / np.logical_or(mask > 0, remask > 0).sum()
        assert iou > 0.95

    def test_separate_instances_splits_objects(self):
        region = Region(0, 0, 200, 200, 1.0)
        annotations = [
            Annotation(exterior=square(10, 10, 30), label="N"),
            Annotation(exterior=square(120, 120, 30), label="N"),
        ]
        mask = annotations_to_mask(annotations, region, {"N": 1})

        split = mask_to_annotations(mask, region, {"N": 1}, separate_instances=True, min_area=5)
        merged = mask_to_annotations(mask, region, {"N": 1}, separate_instances=False, min_area=5)
        assert len(split) == 2
        assert len(merged) == 2  # two disconnected components still trace separately

    def test_min_area_drops_specks(self):
        region = Region(0, 0, 100, 100, 1.0)
        annotations = [
            Annotation(exterior=square(10, 10, 40), label="N"),
            Annotation(exterior=square(80, 80, 2), label="N"),
        ]
        mask = annotations_to_mask(annotations, region, {"N": 1})
        assert len(mask_to_annotations(mask, region, {"N": 1}, min_area=100)) == 1

    def test_simplification_reduces_vertices(self):
        region = Region(0, 0, 200, 200, 1.0)
        circle = np.array(
            [[100 + 60 * math.cos(t), 100 + 60 * math.sin(t)] for t in np.linspace(0, 2 * math.pi, 200)]
        )
        mask = annotations_to_mask([Annotation(exterior=circle, label="N")], region, {"N": 1})

        raw = mask_to_annotations(mask, region, {"N": 1}, simplify_tolerance=0.0)[0]
        simplified = mask_to_annotations(mask, region, {"N": 1}, simplify_tolerance=1.5)[0]
        assert len(simplified.exterior) < len(raw.exterior)
        assert len(simplified.exterior) >= 3

    def test_empty_mask_yields_nothing(self):
        region = Region(0, 0, 50, 50, 1.0)
        assert mask_to_annotations(np.zeros((50, 50), np.uint8), region, {"N": 1}) == []

    def test_accepts_leading_channel_dim(self):
        region = Region(0, 0, 100, 100, 1.0)
        mask = annotations_to_mask([Annotation(exterior=square(20, 20, 40), label="N")], region, {"N": 1})
        assert len(mask_to_annotations(mask[None], region, {"N": 1}, min_area=5)) == 1


class TestGeoJsonIO:
    def test_file_round_trip(self, tmp_path):
        annotations = [Annotation(exterior=square(10, 20, 30), label="Nucleus")]
        geojson = annotations_to_geojson(annotations, colors={"Nucleus": [255, 0, 0]})

        path = tmp_path / "out.geojson"
        save_geojson(geojson, path)
        reloaded = parse_annotations(load_geojson(path))

        assert len(reloaded) == 1
        assert reloaded[0].label == "Nucleus"
        np.testing.assert_allclose(reloaded[0].exterior[0], [10, 20])

    def test_rings_are_closed(self):
        geojson = annotations_to_geojson([Annotation(exterior=square(0, 0, 10), label="N")])
        ring = geojson["features"][0]["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]

    def test_parses_qupath_classification_variants(self):
        features = [
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [square(0, 0, 5).tolist()]},
             "properties": {"classification": {"name": "Modern"}}},
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [square(0, 0, 5).tolist()]},
             "properties": {"classification": "BareString"}},
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [square(0, 0, 5).tolist()]},
             "properties": {"name": "LegacyName"}},
        ]
        labels = [a.label for a in parse_annotations({"type": "FeatureCollection", "features": features})]
        assert labels == ["Modern", "BareString", "LegacyName"]

    def test_accepts_bare_list_and_single_feature(self, tmp_path):
        import json

        feature = {"type": "Feature", "geometry": {"type": "Polygon",
                   "coordinates": [square(0, 0, 5).tolist()]}, "properties": {}}

        for payload in ([feature], feature):
            path = tmp_path / "x.geojson"
            path.write_text(json.dumps(payload))
            assert len(parse_annotations(load_geojson(path))) == 1

    def test_multipolygon_becomes_several_annotations(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "MultiPolygon",
                             "coordinates": [[square(0, 0, 10).tolist()], [square(50, 50, 10).tolist()]]},
                "properties": {"classification": {"name": "N"}},
            }],
        }
        assert len(parse_annotations(geojson)) == 2

    def test_zero_area_geometries_are_ignored(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 2]}, "properties": {}},
                {"type": "Feature", "geometry": {"type": "Polygon",
                 "coordinates": [square(0, 0, 5).tolist()]}, "properties": {}},
            ],
        }
        assert len(parse_annotations(geojson)) == 1

    def test_keep_labels_filters(self):
        geojson = annotations_to_geojson([
            Annotation(exterior=square(0, 0, 5), label="Nucleus"),
            Annotation(exterior=square(20, 20, 5), label="Debris"),
        ])
        assert len(parse_annotations(geojson, keep_labels=["Nucleus"])) == 1
