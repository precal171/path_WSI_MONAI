#!/usr/bin/env bash
#SBATCH --job-name=wsi-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#
# Resource directives come from slurm/config.env; pass them on the command line
# or uncomment and edit these:
##SBATCH --partition=GPUp4
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8
##SBATCH --mem=64G
##SBATCH --time=24:00:00
#
# Full (non-interactive) training run of the nuclei bundle.
#
#     mkdir -p logs
#     sbatch --partition=<PARTITION> --gres=gpu:1 --time=24:00:00 \
#            slurm/train_bundle.sbatch
#
# Override anything at submit time via the environment:
#     MAX_EPOCHS=400 PATCH_DIR=/scratch/$USER/patches sbatch slurm/train_bundle.sbatch
#
# This runs exactly the same configs/train.json that interactive fine-tuning in
# QuPath uses -- only the epoch count and batch size differ. See
# docs/bundle_design.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "$SCRIPT_DIR/config.env" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.env"
else
    echo "ERROR: slurm/config.env not found. cp slurm/config.env.example slurm/config.env" >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$PROJECT_ROOT}"
CONTAINER="${CONTAINER:-$PROJECT_DIR/path_wsi_monai.sif}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
BUNDLE="${BUNDLE:-$PROJECT_DIR/bundles/nuclei_segmentation}"
PATCH_DIR="${PATCH_DIR:-$PROJECT_DIR/data/patches}"
MANIFEST="${MANIFEST:-$PATCH_DIR/manifest.json}"

[[ -n "${APPTAINER_MODULE:-}" ]] && { module load "$APPTAINER_MODULE" 2>/dev/null || true; }

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-$PROJECT_DIR/logs}" "$BUNDLE/models"

if [[ ! -f "$MANIFEST" ]]; then
    cat >&2 <<EOF
ERROR: no training manifest at $MANIFEST

Build one from your QuPath annotations first:

    python tools/prepare_training_data.py \\
        --input-dir "$WSI_DIR" \\
        --output-dir "$PATCH_DIR" \\
        --labels ${LABELS:-Nucleus} \\
        --roi-labels ROI

(or --slide/--annotations for a single slide). Note --roi-labels: without it,
unannotated objects inside a patch are taught to the model as background.
EOF
    exit 1
fi

echo "=============================================================="
echo " Training  job=${SLURM_JOB_ID:-interactive}  node=$(hostname -s)"
echo " bundle    : $BUNDLE"
echo " manifest  : $MANIFEST"
echo " epochs    : ${MAX_EPOCHS:-200}   batch: ${BATCH_SIZE:-16}   lr: ${LEARNING_RATE:-0.0001}"
echo " GPU       : ${CUDA_VISIBLE_DEVICES:-none allocated}"
echo "=============================================================="

BIND_ARG=""
[[ -n "${BIND_PATHS:-}" ]] && BIND_ARG="--bind $BIND_PATHS"

# `finetune=true` continues from models/model.pt. Off by default here: a full
# batch run is usually meant to retrain properly, whereas interactive updates
# are the incremental path. Set FINETUNE=true to continue instead.
run_training() {
    "$@" python -m monai.bundle run \
        --config_file "$BUNDLE/configs/train.json" \
        --logging_file "$BUNDLE/configs/logging.conf" \
        --bundle_root "$BUNDLE" \
        --manifest "$MANIFEST" \
        --ckpt_dir "$BUNDLE/models" \
        --max_epochs "${MAX_EPOCHS:-200}" \
        --batch_size "${BATCH_SIZE:-16}" \
        --learning_rate "${LEARNING_RATE:-0.0001}" \
        --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
        --finetune "${FINETUNE:-false}" \
        --amp true
}

if [[ "${USE_CONDA:-0}" == "1" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate path-wsi-monai
    run_training
else
    [[ -f "$CONTAINER" ]] || { echo "ERROR: no container at $CONTAINER -- run bash slurm/build_container.sh" >&2; exit 1; }
    run_training "$APPTAINER_BIN" exec --nv $BIND_ARG --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER"
fi

echo
echo "Done. Checkpoints in $BUNDLE/models:"
ls -lh "$BUNDLE/models" || true
echo
echo "model.pt is the best-by-validation checkpoint; the MONAI Label server and"
echo "tools/batch_infer.py both pick it up from here with no copying needed."
