"""Regressions for the bugs found in the first code review of this package.

Each test here corresponds to a defect that shipped and was fixed; the comment
on each says what used to happen, so a future change that reintroduces the
behaviour fails with an explanation rather than a bare assertion.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# main.py is exec'd as a top-level module by MONAI Label's app loader
# ---------------------------------------------------------------------------


def _load_main_as_toplevel():
    """Load main.py the way MONAI Label does: exec'd, with no parent package."""
    spec = importlib.util.spec_from_file_location(
        "monai_pathology_main_toplevel", os.path.join(REPO_ROOT, "monai_pathology", "main.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_loads_without_parent_package():
    """Relative imports raised ImportError here, so the server never started."""
    module = _load_main_as_toplevel()
    assert "nuclei_segmentation" in module.AVAILABLE_TASKS


def test_models_all_skips_unfinished_tasks(tmp_path):
    """`--conf models all` instantiated scaffolded tasks, whose init() raises."""
    module = _load_main_as_toplevel()
    studies = tmp_path / "studies"
    studies.mkdir()

    app = module.MyApp(app_dir=str(tmp_path), studies=str(studies), conf={"models": "all"})

    assert set(app.tasks) == {"nuclei_segmentation"}
    assert module.SCAFFOLDED_TASKS.isdisjoint(app.tasks)


def test_naming_a_scaffolded_task_explains_itself(tmp_path):
    """Naming one directly should still fail -- but say why, not NotImplementedError."""
    module = _load_main_as_toplevel()
    studies = tmp_path / "studies"
    studies.mkdir()

    with pytest.raises(NotImplementedError, match="scaffolded but not finished"):
        module.MyApp(
            app_dir=str(tmp_path), studies=str(studies), conf={"models": "tissue_segmentation"}
        )


# ---------------------------------------------------------------------------
# Slide tiling has to reach the right and bottom edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slide_size,patch",
    [((1000, 1000), 256), ((513, 999), 256), ((300, 260), 256)],
)
def test_tiling_covers_the_far_edge(slide_size, patch):
    """The last row/column was dropped, leaving up to patch-1 px never inferred."""
    from monai_pathology.lib.utils.wsi_utils import tile_regions

    regions = list(tile_regions(slide_size, patch_size=patch))
    assert regions, "expected at least one tile"

    assert max(r.x + r.width for r in regions) == slide_size[0]
    assert max(r.y + r.height for r in regions) == slide_size[1]
    assert all(r.width == patch and r.height == patch for r in regions), "tiles must stay uniform"


def test_tiling_does_not_duplicate_on_exact_multiples():
    """The edge fix must not add a redundant tile when the size divides evenly."""
    from monai_pathology.lib.utils.wsi_utils import tile_regions

    regions = list(tile_regions((1024, 1024), patch_size=256))
    assert len(regions) == 16
    assert len({(r.x, r.y) for r in regions}) == 16


def test_tiling_skips_areas_smaller_than_one_patch():
    from monai_pathology.lib.utils.wsi_utils import tile_regions

    assert list(tile_regions((100, 100), patch_size=256)) == []


# ---------------------------------------------------------------------------
# Instance extraction: correctness and memory
# ---------------------------------------------------------------------------


def test_instances_keep_holes_and_absolute_position():
    """Cropping each component to its bbox must not shift or fill its geometry."""
    from monai_pathology.lib.utils.geojson_utils import Region, mask_to_annotations

    mask = np.zeros((60, 60), np.uint8)
    mask[5:20, 5:20] = 1  # plain square
    mask[30:55, 30:55] = 1
    mask[38:47, 38:47] = 0  # square with a hole

    region = Region(x=100, y=200, width=60, height=60, downsample=1.0)
    annotations = mask_to_annotations(
        mask, region, label_map={"Nucleus": 1}, min_area=0, simplify_tolerance=0.0
    )

    assert len(annotations) == 2
    assert sorted(len(a.interiors) for a in annotations) == [0, 1]

    # Coordinates must be absolute (region origin applied), not bbox-relative.
    by_size = sorted(annotations, key=lambda a: len(a.exterior))
    xs = [p[0] for p in by_size[0].exterior]
    ys = [p[1] for p in by_size[0].exterior]
    assert min(xs) == 105 and max(xs) == 119
    assert min(ys) == 205 and max(ys) == 219


def test_dense_tile_does_not_allocate_per_object_masks():
    """One full-tile bool array per nucleus put a dense 512^2 tile at ~131 MB."""
    import tracemalloc

    from monai_pathology.lib.utils.geojson_utils import Region, mask_to_annotations

    rng = np.random.default_rng(1)
    mask = np.zeros((512, 512), np.uint8)
    yy, xx = np.mgrid[0:512, 0:512]
    for _ in range(500):
        cy, cx = rng.integers(8, 504), rng.integers(8, 504)
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 < 16] = 1

    region = Region(x=0, y=0, width=512, height=512, downsample=1.0)

    tracemalloc.start()
    annotations = mask_to_annotations(mask, region, label_map={"Nucleus": 1}, min_area=0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(annotations) > 100
    # Generous bound: the old code peaked around 131 MB here.
    assert peak < 40_000_000, f"peak {peak / 1e6:.0f} MB suggests per-object full-tile masks"


# ---------------------------------------------------------------------------
# Checkpoint loading diagnostics
# ---------------------------------------------------------------------------


def _bundle_utils():
    spec = importlib.util.spec_from_file_location(
        "bundle_scripts_utils",
        os.path.join(REPO_ROOT, "bundles", "nuclei_segmentation", "scripts", "utils.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mismatched_checkpoint_names_the_bad_keys(tmp_path):
    """strict=True raised a bare torch RuntimeError before the diagnostic ran."""
    utils = _bundle_utils()
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1))

    path = tmp_path / "wrong.pt"
    torch.save({"model": {"0.weight": torch.zeros(9, 9, 1, 1), "bogus": torch.zeros(1)}}, path)

    with pytest.raises(RuntimeError) as excinfo:
        utils.load_weights(net, path, "cpu")

    message = str(excinfo.value)
    assert "bogus" in message, "the unexpected key should be named"
    assert "0.weight" in message and "shape mismatch" in message, "shape clash should be named"
    assert "task_type" in message, "should suggest what usually causes this"


def test_wrong_num_classes_checkpoint_is_explained(tmp_path):
    """The realistic case: weights trained with a different output channel count."""
    import sys

    bundle_root = os.path.join(REPO_ROOT, "bundles", "nuclei_segmentation")
    if bundle_root not in sys.path:
        sys.path.insert(0, bundle_root)
    from scripts.network import get_network

    utils = _bundle_utils()
    path = tmp_path / "four_class.pt"
    torch.save({"model": get_network("tissue_segmentation", num_classes=4).state_dict()}, path)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        utils.load_weights(get_network("nuclei_segmentation", num_classes=1), path, "cpu")

    # Non-strict should degrade gracefully rather than explode.
    net = get_network("nuclei_segmentation", num_classes=1)
    utils.load_weights(net, path, "cpu", strict=False)


def test_non_strict_load_warns_and_continues(tmp_path, caplog):
    import logging

    utils = _bundle_utils()
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1))

    path = tmp_path / "partial.pt"
    torch.save({"model": {"0.weight": net[0].weight.detach().clone()}}, path)  # no bias

    with caplog.at_level(logging.WARNING):
        utils.load_weights(net, path, "cpu", strict=False)

    assert "does not match" in caplog.text
    assert "0.bias" in caplog.text, "the missing key should be named"


def test_missing_checkpoint_is_not_silently_ignored(tmp_path):
    utils = _bundle_utils()
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1))

    with pytest.raises(FileNotFoundError, match="Train the model first"):
        utils.load_weights(net, tmp_path / "nope.pt", "cpu")


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def _infer_task_with_weights(tmp_path, roi_size):
    """A NucleiSegmentationInfer pointed at a real (untrained) checkpoint."""
    import sys

    bundle_root = os.path.join(REPO_ROOT, "bundles", "nuclei_segmentation")
    if bundle_root not in sys.path:
        sys.path.insert(0, bundle_root)
    from scripts.network import get_network

    ckpt = tmp_path / "model.pt"
    torch.save({"model": get_network("nuclei_segmentation").state_dict()}, ckpt)

    from monai_pathology.lib.infers.nuclei_segmentation import NucleiSegmentationInfer

    return NucleiSegmentationInfer(
        path=str(ckpt), bundle_root=bundle_root, roi_size=roi_size
    )


def test_patch_size_reaches_the_inferer(tmp_path):
    """--conf patch_size set self.roi_size but left the sliding window at 256."""
    task = _infer_task_with_weights(tmp_path, roi_size=(128, 128))
    pieces = task._load_pieces(torch.device("cpu"))

    assert tuple(pieces["inferer"].roi_size) == (128, 128)


def test_infer_pieces_are_cached_per_device(tmp_path):
    """The cache ignored device, so a later cpu request got the cuda network."""
    task = _infer_task_with_weights(tmp_path, roi_size=(256, 256))

    first = task._load_pieces(torch.device("cpu"))
    again = task._load_pieces(torch.device("cpu"))
    assert again is first, "same device should reuse the cached network"

    # A cache entry for another device must never be handed back for this one.
    sentinel = {"network": object()}
    task._pieces["cuda:0"] = sentinel
    assert task._load_pieces(torch.device("cpu")) is first
    assert set(task._pieces) == {"cpu", "cuda:0"}


def test_server_port_variable_matches_the_script():
    """config.env.example set MONAILABEL_PORT; the script read a different name."""
    with open(os.path.join(REPO_ROOT, "slurm", "config.env.example")) as handle:
        example = handle.read()
    with open(os.path.join(REPO_ROOT, "slurm", "start_label_server.sbatch")) as handle:
        script = handle.read()

    assert "MONAILABEL_PORT=" in example
    assert "${MONAILABEL_PORT:-" in script, "the script must read the name config.env sets"
    assert "MONAILABEL_PORT_OVERRIDE" not in script + example


def test_tools_package_is_installed():
    """tools/ is imported at runtime by the trainer, so it must ship."""
    import tomllib

    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as handle:
        config = tomllib.load(handle)

    include = config["tool"]["setuptools"]["packages"]["find"]["include"]
    assert any(entry.startswith("tools") for entry in include)


# ---------------------------------------------------------------------------
# Multi-class annotations must not vanish quietly
# ---------------------------------------------------------------------------


def test_multiclass_binary_mode_warns(caplog):
    """--labels A B C built a multi-value map, then wrote (mask > 0) silently."""
    import logging

    from tools.prepare_training_data import _resolve_mask_mode

    with caplog.at_level(logging.WARNING):
        assert _resolve_mask_mode("binary", ["Tumour", "Stroma", "Necrosis"]) == "binary"

    assert "merges them all into one foreground" in caplog.text


def test_single_class_binary_mode_is_quiet(caplog):
    import logging

    from tools.prepare_training_data import _resolve_mask_mode

    with caplog.at_level(logging.WARNING):
        _resolve_mask_mode("binary", ["Nucleus"])

    assert caplog.text == ""


def test_class_index_mode_warns_about_the_bundle(caplog):
    import logging

    from tools.prepare_training_data import _resolve_mask_mode

    with caplog.at_level(logging.WARNING):
        _resolve_mask_mode("class-index", ["Tumour", "Stroma"])

    assert "multi-class bundle" in caplog.text


def test_bad_mask_mode_is_rejected():
    from tools.prepare_training_data import _resolve_mask_mode

    with pytest.raises(SystemExit, match="mask-mode"):
        _resolve_mask_mode("rgb", ["Nucleus"])
