# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A whole-slide pathology pipeline for UTSW BioHPC: annotate slides in QuPath, train from those annotations via a MONAI Label server on a GPU node, apply the model across a slide archive with SLURM job arrays. Everything runs on the cluster; the code is bind-mounted into a pulled container, never baked in.

## Commands

```bash
python -m pytest tests/                    # full suite, ~3 min (includes a CPU training run)
python -m pytest tests/ -m "not slow"      # ~10 s
python -m pytest tests/test_biohpc_layer.py -q   # just the deployment-layer regressions
```

Tests need CPU torch + MONAI but no GPU, slides, QuPath, or `monailabel` (app-layer tests skip cleanly without it). Shell changes: `bash -n slurm/*.sh slurm/*.sbatch` and the suite's `test_every_shell_script_parses`.

**Setting up on BioHPC** (the task this repo exists for) — follow `docs/biohpc_setup.md` in order. The short form, run on a login node from `/project` or `/work` space (not `$HOME` — the image is ~10 GB):

```bash
cp slurm/config.env.example slurm/config.env   # then edit: WSI_DIR + partition minimum
bash slurm/build_container.sh                  # pulls pinned MONAI image + layers a venv; no root
srun --partition=GPUA100 --gres=gpu:1 --pty bash slurm/check_env.sh   # must be all PASS
bash slurm/submit.sh server                    # then: cat logs/server_<jobid>.out
```

`check_env.sh` is the gate: do not debug anything downstream until every line passes.

## Hard-won constraints (violating these re-breaks fixed bugs)

- **Never `sbatch slurm/*.sbatch` directly and never document doing so.** `#SBATCH` directives are parsed before the shell runs, so the .sbatch files cannot read `config.env`; all resource flags come from `slurm/submit.sh`. A bare sbatch submits with no partition and no GPU. A test enforces that docs only show `submit.sh`.
- **Every variable a `slurm/` script reads must be defined in `config.env.example`** — `tests/test_biohpc_layer.py` enforces coverage and that the example sources cleanly under `set -u` (definition order matters: `PROJECT_DIR` before anything that expands it).
- **The base image is pinned** (`projectmonai/monai:1.4.0`, CUDA 12.x, targets A100/H100). Never `:latest`; never an unconstrained torch reinstall in `apptainer.def` — pip is constrained to the image's own torch. Pascal-era GPUp4 does not work with recent torch.
- **`monai_pathology/main.py` is exec'd as a top-level module** by MONAI Label's app loader — absolute imports only, via its `sys.path` bootstrap. Relative imports break the server.
- **`--conf models all` must skip `SCAFFOLDED_TASKS`** (tissue_segmentation, tile_classification — their `init()` raises by design).
- **One bundle, two callers:** interactive fine-tune (QuPath Train button, `lib/trainers/`) and batch training (`submit.sh train`) both run `bundles/nuclei_segmentation/configs/train.json`. Do not fork the training path.
- **The repo has no CI.** Run the suite before pushing; it is the only gate.

## Architecture

Three layers, coupled only through files on cluster storage:

1. **`monai_pathology/`** — the MONAI Label app QuPath talks to over REST. `lib/_compat.py` isolates base-class import paths that move between monailabel releases (fix version bumps there, one file). Tasks register in `AVAILABLE_TASKS` in `main.py`; each is a `TaskConfig` in `lib/configs/` wiring an infer task (`lib/infers/`) and trainer (`lib/trainers/`). Adding a task type = one config module + one bundle, nothing else.
2. **`bundles/nuclei_segmentation/`** — standard MONAI Bundle (BasicUNet, DiceCE, sliding-window inference). Weights live in `models/model.pt`, read and written by the server, batch training, and batch inference alike — that shared path is what makes an overnight training run "live" in QuPath after a server restart, with no copying.
3. **`slurm/` + `tools/`** — cluster entry points (`submit.sh` → three .sbatch scripts; `build_container.sh`; `check_env.sh` preflight) and CLIs (`tools/prepare_training_data.py` GeoJSON→patches, `tools/batch_infer.py` slide→GeoJSON, `tools/build_synthetic_wsi.py` test fixture). `tools/` is imported at runtime by the trainer, so it ships in the package.

Coordinate interchange lives in one place: `monai_pathology/lib/utils/geojson_utils.py` (QuPath GeoJSON polygons in level-0 (x,y) ↔ rasterized masks, offset/downsample math) and `wsi_utils.py` (patch reads; note MONAI's WSIReader uses (y,x) order). All annotation and prediction geometry flows through these two files.

Runtime environment: a pulled `.sif` plus a `--system-site-packages` venv on shared storage (`VENV_DIR`), which keeps the image's CUDA-matched torch and adds project deps (incl. the `openslide-bin` wheel that supplies libopenslide without apt). Scripts run `$VENV_DIR/bin/...` inside the container when `VENV_DIR` is set; an empty `VENV_DIR` means a self-contained image built with `build_container.sh --build-def`.

## Testing philosophy

Assert observable facts — absolute coordinates, IoU against ground truth, named keys in error messages — not that a function returned. `tests/test_end_to_end.py` (synthetic slide → train → infer → GeoJSON, scored) is the test that catches self-consistent-but-wrong coordinate bugs; keep it passing. `tests/test_review_regressions.py` and `tests/test_biohpc_layer.py` each pin a shipped defect with a comment saying what used to happen — extend them in kind when fixing bugs.
