# Architecture

## Layers

```
QuPath  ──REST──▶  MONAI Label server          monai_pathology/
                     ├── infers/               region -> GeoJSON
                     ├── trainers/             annotations -> fine-tune
                     ├── activelearning/       which slide next
                     └── configs/              per-task wiring
                              │
                              ▼
                   MONAI Bundle                bundles/nuclei_segmentation/
                     network, transforms, train + inference configs
                              │
                              ▼
                   WSI + GeoJSON utilities     monai_pathology/lib/utils/
                     the QuPath <-> MONAI contract
```

Dependencies point downward only. `lib/utils/` knows nothing about MONAI Label
and can be used — and tested — without it.

## Decisions worth explaining

### The version-drift shim (`lib/_compat.py`)

MONAI Label has moved `TaskConfig`, `BasicInferTask`, `BasicTrainTask` and
`Strategy` between `monailabel.interfaces.*` and `monailabel.tasks.*` across
releases. Importing them directly from a guessed path means a version bump
breaks every module at once, with a traceback pointing at whichever file Python
loaded first rather than at the real problem.

One module resolves them, trying known locations in order, and raises a single
error naming what to check. It also provides placeholders when `monailabel` is
absent, so the package imports in an environment that only does training and
batch inference — and so the test suite runs there.

Paths verified against monailabel 0.8.x, pinned in `tests/test_app_imports.py`.

### Hard logic kept out of the MONAI Label surface

`NucleiSegmentationInfer.infer_region(slide_path, region)` takes no MONAI Label
types and returns GeoJSON. `__call__` is a thin adapter that unpacks the request
dict and calls it.

The request/response contract with QuPath is the most version-fragile part of
this project. Keeping the segmentation logic behind a plain interface means a
protocol change is confined to one small method, and the interesting code stays
testable without a server, a GPU or the package installed.

### One module owns the coordinate maths

`lib/utils/geojson_utils.py` handles both directions of the QuPath interchange —
polygons to masks and masks to polygons.

Three conventions collide here:

| | Order | Space |
|---|---|---|
| QuPath GeoJSON | `(x, y)` | level-0 slide pixels |
| NumPy mask | `[row, col]` = `[y, x]` | patch-local |
| `WSIReader.get_data` | `location=(row, col)` | level-0, but `size` is at the read level |

Get any of that wrong and patches and masks describe different parts of the
slide. Nothing raises — the model simply never learns. `Region` carries the
offset and downsample, both directions live in one module, and
`tests/test_geojson_conversion.py` asserts absolute coordinates rather than
"something came back".

Note the asymmetry in `WSIReader`: `location` is always level-0 while `size` is
at the level being read. `read_region` derives both from one `Region` so callers
cannot mix them, and resizes when no pyramid level matches the requested
downsample — without which a downsampled read silently returns a full-resolution
crop of the wrong area.

### `tools/`, not `scripts/`

Every MONAI bundle has a `scripts/` package on `sys.path`. A top-level
`scripts/` package would collide: Python allows one `sys.modules['scripts']`, so
whichever path came first would win, and the trainer's
`from scripts.prepare_training_data import ...` would resolve to the bundle's
package instead. Renaming removes the ambiguity.

### Random active learning by default

Uncertainty sampling on an untrained model is worse than random: the model's
confusion is noise, so it directs annotation effort arbitrarily. Random is the
baseline any strategy must beat. `epistemic.py` documents the three conditions
that must hold first — dropout enabled, a model worth listening to, and the
compute budget for N forward passes per candidate region.

## Extending

Adding a task type touches two places: a `TaskConfig` in `lib/configs/`, and its
name in `main.py`'s `AVAILABLE_TASKS`. Nothing in the server wiring, the data
pipeline or the SLURM scripts is task-specific.

Already generic: `scripts/network.py` (three task types), `geojson_utils`
(multi-class masks natively), `prepare_training_data.py` (multiple `--labels`).

Task-specific and needing work per task: the bundle configs (loss and
postprocessing differ between binary, multi-class and classification), and mask
handling in data prep — currently binarised, which multi-class needs to preserve.

`lib/configs/tissue_segmentation.py` and `tile_classification.py` spell out what
remains for each, and raise on `init()` rather than half-working.

## Deliberately not built

- **NuClick** (click-to-segment). Its click-conditioning API is the piece I could
  least verify; better taken from the official sample app than reconstructed.
- **MC-dropout uncertainty.** Needs a trained dropout-enabled model first.
- **Digital Slide Archive datastore.** MONAI Label supports it; the filesystem
  datastore is simpler and sufficient for a single-lab workflow.
- **Stain normalisation.** MONAI has `NormalizeHEStains`. Worth adding when you
  train across scanners or sites; unnecessary within one.
