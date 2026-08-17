# Testing

```bash
python -m pytest tests/                 # everything, ~45s
python -m pytest tests/ -m "not slow"   # skip the CPU training run, ~10s
```

130 tests. None needs a GPU, a real slide, QuPath, or a running server.

## What the suite actually covers

| File | Covers |
|---|---|
| `test_geojson_conversion.py` | Coordinate maths in both directions, at absolute coordinates, with offsets, downsampling, holes, multi-class and the QuPath dialects |
| `test_wsi_patch_extraction.py` | `WSIReader` `(y,x)` vs GeoJSON `(x,y)`, downsampled reads, tiling, tissue detection |
| `test_bundle_config.py` | Every `@reference` and `$expression` resolves; the trainer runs; inference normalisation matches training |
| `test_app_imports.py` | Every module imports with and without `monailabel`; the compat shim resolves the real classes; subclasses construct |
| `test_end_to_end.py` | Slide → patches → training → inference → GeoJSON, scored against ground truth |

The end-to-end test is the valuable one. It trains a real model on a synthetic
slide and checks the predictions land on the actual nuclei, which is what catches
integration bugs that leave every unit test green: a self-consistent but wrong
coordinate convention, a normalisation mismatch, an inferer configured wider than
the tiles it is fed.

Two real bugs were caught this way during development, both silent:

- A downsampled `read_region` returned a full-resolution crop of the **wrong
  area** whenever no pyramid level matched exactly.
- A model scoring **0.964 dice** produced **zero** detected objects, because the
  sliding window (256) was wider than the tiles (128), so MONAI padded every tile
  and the predictions never crossed the threshold. Inference "succeeded" and
  wrote empty GeoJSON.

Both now have regression tests.

## What it does not prove

Green tests here say the pipeline is *wired* correctly. They say nothing about
these, all of which need your cluster and your data:

- **Real slide formats.** Everything is tested against synthetic pyramidal TIFFs
  read with the `TiffFile` backend. Actual `.svs`/`.ndpi` via OpenSlide, with
  vendor metadata, odd pyramid layouts and JPEG2000 tiles, is untested here.
- **The container.** `slurm/apptainer.def` has never been built in this
  environment. Its `%test` section checks imports at build time — watch it.
- **GPU.** No CUDA anywhere. `apptainer exec --nv` GPU visibility is
  step 3 of [biohpc_setup.md](biohpc_setup.md) for a reason.
- **SLURM.** The scripts are syntax-checked (`bash -n`) and no more. Partitions,
  accounts, module names and mounts are all site-specific.
- **QuPath.** The extension, the REST protocol, and the request/response shapes
  are unverified. `_region_from_request` is tested against what the protocol is
  *believed* to send. If QuPath shows no annotations, log the request dict first —
  the geometry is usually right and the wrapping is what differs.
- **MONAI Label end to end.** Class resolution and subclass construction are
  tested; a live server, datastore persistence and the annotation round trip are
  not.
- **Model quality.** The e2e thresholds (recall/precision > 0.5) prove wiring,
  not usefulness. Real performance depends entirely on your annotations.

## Your first run on BioHPC

Work in this order — each step's failure is unambiguous, which stops you
debugging three things at once:

1. `apptainer exec --nv ... torch.cuda.is_available()` → must print `True`.
2. `python -m pytest tests/` **inside the container** → the suite should pass
   there too. If it fails only in the container, it is the image, not the code.
3. Read one real slide:
   ```python
   from monai_pathology.lib.utils.wsi_utils import get_slide_info
   print(get_slide_info("/path/to/real.svs", backend="OpenSlide"))
   ```
   Sane dimensions, several levels, and ideally an `mpp` value.
4. The synthetic walkthrough in the README, end to end, inside the container.
5. Start the server; confirm QuPath connects and lists slides.
6. Annotate one region, train, run inference on it. Predictions should land on
   tissue — that is the first evidence the QuPath coordinate contract is right.

Only then annotate in earnest.

## Adding tests

Assert observable facts, not that a function returned. `assert len(result) > 0`
would have passed while both silent bugs above were live. Prefer absolute
coordinates, IoU against a known mask, or a metric threshold.
