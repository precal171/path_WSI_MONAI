# One bundle, three callers

The MONAI Bundle in `bundles/nuclei_segmentation/` is the single definition of
what the model is. Three very different callers use it, and the design goal is
that none of them can drift away from the others.

```
configs/train.json ────┬── monai_pathology/lib/trainers/  (interactive fine-tune)
                       └── slurm/train_bundle.sbatch      (full batch training)

configs/inference.json ┬── monai_pathology/lib/infers/    (interactive, QuPath)
                       ├── tools/batch_infer.py           (slide-wide, SLURM)
                       └── monai.bundle run               (folder of patches)
```

## Why it matters

The obvious alternative — a quick training path for interactive use and a
"proper" one for batch — fails in a specific and expensive way. The user
annotates, clicks Train, sees the model improve in QuPath, then runs an
overnight job and gets a different model. Now every improvement is ambiguous:
was that the annotations, or a difference in augmentation, normalisation or
loss between two code paths? Debugging that costs days.

Here `train.json` is loaded by both. The interactive trainer overrides
`max_epochs`, `batch_size` and `finetune`, and changes nothing else.

## Training

`monai_pathology/lib/trainers/nuclei_segmentation.py` builds a manifest from the
datastore's annotations and then does the same thing the sbatch script does:

```python
parser = ConfigParser()
parser.read_config(f"{bundle_root}/configs/train.json")
parser.update({"manifest": ..., "max_epochs": 20, "finetune": True})
for statement in parser.get_parsed_content("run"):
    ...
```

```bash
python -m monai.bundle run \
    --config_file bundles/nuclei_segmentation/configs/train.json \
    --bundle_root bundles/nuclei_segmentation \
    --manifest data/patches/manifest.json \
    --max_epochs 200
```

`BasicTrainTask` also declares abstract `network()`, `optimizer()`,
`loss_function()` and friends. They are implemented, and each resolves from the
same `train.json`, so anything that inspects the trainer sees exactly what
training used rather than a second declaration that can rot.

## Inference

`inference.json` exposes its transform chain twice, because patches arrive two
different ways:

- `preprocessing` — loads from a file path. Used by the standalone
  `monai.bundle run` path.
- `array_preprocessing` — takes an in-memory `CHW` array. Used by
  `batch_infer.py` and the MONAI Label infer task, whose patches come from a WSI
  reader and were never files.

Both are built from one `array_transforms` list, so the normalisation cannot
differ between them. `tests/test_bundle_config.py` asserts that inference
normalisation matches training's.

## The manifest

`tools/prepare_training_data.py` writes the contract both training paths consume:

```json
{
  "task": "nuclei_segmentation",
  "labels": ["Nucleus"],
  "training":   [{"image": "/abs/img_x0_y0.png", "label": "/abs/msk_x0_y0.png", "slide": "s1"}],
  "validation": [...]
}
```

Absolute paths, because on a cluster the patch directory is stable and
regenerating the manifest is cheap. Masks are stored as 0/255 PNGs so you can
eyeball them; `train.json` rescales with fixed (not data-derived) ranges, so an
all-background patch cannot divide by zero.

Validation is split **by slide** whenever more than one slide is present.
Splitting patches from one slide across train and validation leaks — adjacent
tiles share tissue, staining and scanner characteristics — and the resulting
scores are optimistic. With one slide the script splits by patch and warns.

## Two traps worth knowing about

**`_comment` inside a component.** MONAI strips only its own `_`-prefixed keys
(`_target_`, `_desc_`, `_requires_`, `_disabled_`, `_mode_`). Any other
underscore key inside a `_target_` dict is passed to the constructor as a
keyword argument and raises. Use `_desc_`. There is a test for this.

**`roi_size` larger than the tile.** If the `SlidingWindowInferer`'s `roi_size`
exceeds the patch being fed to it, MONAI pads the patch out to `roi_size` and the
network sees a border of constant padding it never met in training. Predictions
come back systematically under-confident. This is nasty because nothing raises:
during development a model scoring **0.964 dice** produced a maximum probability
of **0.477** and therefore *zero* objects after thresholding at 0.5 — inference
"worked" and returned empty GeoJSON. `load_bundle_pieces(..., patch_size=...)`
now keeps them in step, the interactive task shrinks its window to fit, and
`tests/test_end_to_end.py` guards both.

## Adding a task

1. Copy `bundles/nuclei_segmentation/` and adjust `configs/`.
2. For multi-class, switch the loss to `DiceCELoss(softmax=True, to_onehot_y=True)`
   and postprocessing to `AsDiscreted(argmax=True)`.
3. `scripts/network.py` already handles `tissue_segmentation` and
   `tile_classification`; pass the right `task_type` and `num_classes`.
4. Add a `TaskConfig` in `monai_pathology/lib/configs/` and register it in
   `main.py`'s `AVAILABLE_TASKS`.

The scaffold files in `lib/configs/` list what remains for each.
