# path_WSI_MONAI

A whole-slide pathology pipeline built on NVIDIA MONAI: annotate slides in
**QuPath**, train models from your own annotations with **MONAI Label**'s
active-learning loop, and apply them across a slide archive with **MONAI Core**.

**Everything runs on BioHPC.** Nothing is installed on your own machine — one
Python virtualenv on cluster storage (plain `pip`, no root, no container
runtime) serves the annotation server, training and batch inference. Your
slides never leave cluster storage, and model weights never get copied between
machines. The only thing that runs locally is QuPath itself, and even that can
run inside a BioHPC Web Visualization session
(see [docs/biohpc_setup.md](docs/biohpc_setup.md)).

## The loop

```
   QuPath  ──annotate──▶  MONAI Label server  ──▶  patches + masks
     ▲                    (GPU node, SLURM)              │
     │                                                   ▼
     │                                        bundles/nuclei_segmentation
     │                                          (one config, two callers)
     │                                            │              │
     │                              interactive fine-tune    sbatch training
     │                                            │              │
     │                                            └──▶ models/model.pt ◀──┘
     │                                                        │
     └────── GeoJSON ◀── batch inference (SLURM job array) ◀───┘
```

Each round of annotation improves the model; each improved model pre-annotates
the next slides, so you correct predictions instead of drawing from scratch.

**Not a programmer? Start with [docs/researcher_guide.md](docs/researcher_guide.md)**
— a plain-language, click-by-click guide to annotating and training through the
BioHPC visualization portal, once the one-time setup below has been done.

## Quick start

All of this happens on the cluster.

```bash
ssh <you>@nucleus.biohpc.swmed.edu
cd /project/<your-space>          # NOT $HOME -- the venv is ~10 GB
git clone https://github.com/precal171/path_wsi_monai.git
cd path_wsi_monai

cp slurm/config.env.example slurm/config.env
$EDITOR slurm/config.env          # partition, account, WSI_DIR -- see the TODOs

bash slurm/build_env.sh           # creates the venv + installs everything (no root needed)

# preflight on a GPU node -- the gate for everything else
srun --partition=GPUA100 --gres=gpu:1 --pty bash slurm/check_env.sh

bash slurm/submit.sh server       # submits with the GPU/partition from config.env
cat logs/server_*.out             # prints the node, port and how to connect
```

Then point QuPath at the server and start annotating. Full walkthrough in
[docs/workflow.md](docs/workflow.md).

### Try it without any slides

The pipeline is runnable end to end on synthetic data, which is a good way to
check the environment before committing real annotation effort:

```bash
python tools/build_synthetic_wsi.py --output-dir data/synthetic
python tools/prepare_training_data.py \
    --slide data/synthetic/synthetic_slide.tiff \
    --annotations data/synthetic/synthetic_slide.geojson \
    --output-dir data/patches --labels Nucleus --backend TiffFile
python -m monai.bundle run \
    --config_file bundles/nuclei_segmentation/configs/train.json \
    --bundle_root bundles/nuclei_segmentation \
    --manifest data/patches/manifest.json --max_epochs 20
python tools/batch_infer.py --bundle bundles/nuclei_segmentation \
    --slide data/synthetic/synthetic_slide.tiff \
    --output-dir outputs --backend TiffFile --patch-size 256
```

## Layout

| Path | What it is |
|---|---|
| `monai_pathology/` | The MONAI Label app: infer, train and active-learning tasks |
| `bundles/nuclei_segmentation/` | MONAI Bundle — network, transforms, training and inference configs |
| `tools/` | CLIs: data prep, batch inference, synthetic-slide generator |
| `slurm/` | Environment setup (`build_env.sh`), preflight (`check_env.sh`), and the job entry point (`submit.sh`) |
| `qupath/` | How to connect QuPath to the server |
| `docs/` | Setup, workflow, architecture, bundle design, testing, and the non-technical researcher guide |
| `tests/` | 146 tests, runnable without GPU, slides or QuPath |

## What works, and what doesn't

**Working and tested end to end:** nucleus segmentation — annotation → patch
extraction → training → inference → GeoJSON back into QuPath, plus interactive
fine-tuning and SLURM batch training/inference.

**Scaffolded, not finished:** tissue-region segmentation and tile classification
(`monai_pathology/lib/configs/`) build networks but have no data pipeline; the
click-interactive NuClick model; and MC-dropout uncertainty sampling for active
learning, which falls back to random selection. Each file says what remains.

The architecture is deliberately task-agnostic — adding a task means writing one
config module and one bundle, not touching the app. See
[docs/architecture.md](docs/architecture.md).

## Tests

```bash
python -m pytest tests/                    # everything (~45s)
python -m pytest tests/ -m "not slow"      # skip the CPU training run
```

No GPU, real slide, QuPath install or running server required — see
[docs/testing.md](docs/testing.md) for exactly what that does and does not
prove, and what you still have to verify by hand on BioHPC.

## Licence

No licence file is included. Add one before sharing this: as university research
code it may be subject to institutional IP policy, so that choice is yours rather
than a default worth guessing at.

## Not for clinical use

Research software. Not a medical device, not validated for diagnostic use.
