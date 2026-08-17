# Nuclei segmentation bundle

Binary 2D segmentation of cell nuclei in RGB whole-slide patches.

| | |
|---|---|
| Network | `BasicUNet`, 2D, 3 in / 1 out |
| Input | 256x256 RGB patch, scaled to [0, 1] |
| Output | single-channel probability map, sigmoid, thresholded at 0.5 |
| Loss | `DiceCELoss(sigmoid=True)` |
| Data | patches cut from QuPath annotations by `tools/prepare_training_data.py` |

## Run

```bash
python -m monai.bundle run \
    --config_file configs/train.json --bundle_root . \
    --manifest ../../data/patches/manifest.json --max_epochs 200

python -m monai.bundle run \
    --config_file configs/inference.json --bundle_root . \
    --input_dir sample_patches --output_dir eval
```

For whole slides use `tools/batch_infer.py`, which reuses this bundle's
transforms and network but does its own WSI tiling and GeoJSON output.

## Keys worth knowing

| Key | Meaning |
|---|---|
| `manifest` | training manifest (see `docs/bundle_design.md`) |
| `finetune` | continue from `models/model.pt` instead of random init |
| `patch_size` | must match the tiles actually fed at inference -- see below |
| `threshold` | probability cutoff for a positive pixel |

`models/model.pt` is the best-by-validation checkpoint and the file everything
else loads. `model_final.pt` is the last epoch, saved so a run that never
improves still leaves usable weights.

## Two traps

`_comment` inside a `_target_` dict is passed to the constructor as a keyword
argument and raises. Use `_desc_`.

If the `SlidingWindowInferer`'s `roi_size` is wider than the patch fed to it,
MONAI pads the patch and the network sees padding it never trained on;
predictions come back under-confident enough to disappear at the threshold, with
no error. Keep `patch_size` in step with the tile size.

Both are covered in `docs/bundle_design.md` and guarded by tests.
