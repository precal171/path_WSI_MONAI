#!/usr/bin/env bash
#SBATCH --job-name=wsi-infer
#SBATCH --output=logs/infer_%A_%a.out
#SBATCH --error=logs/infer_%A_%a.err
#SBATCH --array=0-3
#
# Resource directives come from slurm/config.env; or uncomment and edit:
##SBATCH --partition=GPUp4
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8
##SBATCH --mem=64G
##SBATCH --time=08:00:00
#
# Apply a trained model across a slide archive, writing one GeoJSON per slide
# for import back into QuPath.
#
#     mkdir -p logs
#     sbatch --array=0-9 --partition=<PARTITION> --gres=gpu:1 \
#            slurm/infer_wsi_batch.sbatch
#
# Runs as a job array: each task takes every Nth slide, so ten array tasks means
# ten slides in flight. Size the array to the number of slides and how many GPUs
# your allocation realistically gets -- `--array=0-9%4` caps it at four at once.
#
# %A is the array job id and %a the task index, so each task logs separately.

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
INPUT_DIR="${INFER_INPUT_DIR:-${WSI_DIR:?Set WSI_DIR in slurm/config.env}}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs}"

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

[[ -n "${APPTAINER_MODULE:-}" ]] && { module load "$APPTAINER_MODULE" 2>/dev/null || true; }

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-$PROJECT_DIR/logs}" "$OUTPUT_DIR"

if [[ ! -f "$BUNDLE/models/model.pt" ]]; then
    echo "ERROR: no trained weights at $BUNDLE/models/model.pt" >&2
    echo "       Train first: sbatch slurm/train_bundle.sbatch" >&2
    exit 1
fi

echo "=============================================================="
echo " Batch inference  job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-interactive}}"
echo " shard     : $SHARD_INDEX of $SHARD_COUNT   node=$(hostname -s)"
echo " slides    : $INPUT_DIR"
echo " output    : $OUTPUT_DIR"
echo " GPU       : ${CUDA_VISIBLE_DEVICES:-none allocated}"
echo "=============================================================="

BIND_ARG=""
[[ -n "${BIND_PATHS:-}" ]] && BIND_ARG="--bind $BIND_PATHS"

run_inference() {
    "$@" python tools/batch_infer.py \
        --bundle "$BUNDLE" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --label "${LABELS:-Nucleus}" \
        --backend "${WSI_BACKEND:-OpenSlide}" \
        --patch-size "${INFER_PATCH_SIZE:-512}" \
        --overlap "${INFER_OVERLAP:-64}" \
        --downsample "${DOWNSAMPLE:-1.0}" \
        --batch-size "${INFER_BATCH_SIZE:-8}" \
        --min-tissue-fraction "${MIN_TISSUE_FRACTION:-0.05}" \
        --shard-index "$SHARD_INDEX" \
        --num-shards "$SHARD_COUNT"
}

if [[ "${USE_CONDA:-0}" == "1" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate path-wsi-monai
    run_inference
else
    [[ -f "$CONTAINER" ]] || { echo "ERROR: no container at $CONTAINER -- run bash slurm/build_container.sh" >&2; exit 1; }
    run_inference "$APPTAINER_BIN" exec --nv $BIND_ARG \
        --bind "$PROJECT_DIR:$PROJECT_DIR" --bind "$INPUT_DIR:$INPUT_DIR" "$CONTAINER"
fi

echo
echo "Shard $SHARD_INDEX done. GeoJSON in $OUTPUT_DIR"
echo "Import into QuPath with: Objects > Import objects, or the script in qupath/README.md"
