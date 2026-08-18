#!/usr/bin/env bash
#SBATCH --job-name=wsi-infer
#SBATCH --output=logs/infer_%A_%a.out
#SBATCH --error=logs/infer_%A_%a.err
#
# DO NOT sbatch this file directly: #SBATCH directives are parsed before the
# shell runs, so nothing here can read config.env -- a bare submission gets NO
# partition, NO GPU and no job array. Submit through the wrapper, which passes
# the resources and the array size (INFER_ARRAY_SIZE) from slurm/config.env:
#
#     bash slurm/submit.sh infer
#
# Apply a trained model across a slide archive, writing one GeoJSON per slide
# for import back into QuPath.
#
# Runs as a job array: each task takes every Nth slide, so ten array tasks means
# ten slides in flight. Size INFER_ARRAY_SIZE to the number of slides and how
# many GPUs your allocation realistically gets; INFER_ARRAY_THROTTLE caps
# concurrency. %A is the array job id and %a the task index, so each task logs
# separately.

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
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
BUNDLE="${BUNDLE:-$PROJECT_DIR/bundles/nuclei_segmentation}"
INPUT_DIR="${INFER_INPUT_DIR:-${WSI_DIR:?Set WSI_DIR in slurm/config.env}}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs}"

# The venv on shared storage is the whole environment; no module loads needed
# at job time -- its bin/ binaries resolve on any node that mounts it.
PYTHON_CMD="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON_CMD" ]]; then
    echo "ERROR: no environment at $VENV_DIR -- run: bash slurm/build_env.sh" >&2
    exit 1
fi

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-$PROJECT_DIR/logs}" "$OUTPUT_DIR"

if [[ ! -f "$BUNDLE/models/model.pt" ]]; then
    echo "ERROR: no trained weights at $BUNDLE/models/model.pt" >&2
    echo "       Train first: bash slurm/submit.sh train" >&2
    exit 1
fi

echo "=============================================================="
echo " Batch inference  job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-interactive}}"
echo " shard     : $SHARD_INDEX of $SHARD_COUNT   node=$(hostname -s)"
echo " slides    : $INPUT_DIR"
echo " output    : $OUTPUT_DIR"
echo " GPU       : ${CUDA_VISIBLE_DEVICES:-none allocated}"
echo "=============================================================="

"$PYTHON_CMD" tools/batch_infer.py \
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

echo
echo "Shard $SHARD_INDEX done. GeoJSON in $OUTPUT_DIR"
echo "Import into QuPath with: Objects > Import objects, or the script in qupath/README.md"
