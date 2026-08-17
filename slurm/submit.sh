#!/usr/bin/env bash
# Submit a pipeline job with the resources from slurm/config.env.
#
#     bash slurm/submit.sh server    # MONAI Label annotation server
#     bash slurm/submit.sh train     # full batch training run
#     bash slurm/submit.sh infer     # batch inference job array
#
# Anything after the role is passed through to sbatch, so one-off overrides
# stay easy:
#
#     bash slurm/submit.sh train --time=48:00:00
#
# Why this script exists: `#SBATCH` directives are parsed by SLURM before the
# shell ever runs, so a script cannot read config.env into its own directives.
# Submitting the .sbatch files bare therefore requests NO partition and NO GPU
# -- the job lands CPU-only on the default queue and everything downstream is
# mysteriously slow or broken. This wrapper turns config.env into explicit
# sbatch flags so that cannot happen.
#
# SUBMIT_DRYRUN=1 prints the sbatch command instead of running it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ROLE="${1:-}"
if [[ "$ROLE" != "server" && "$ROLE" != "train" && "$ROLE" != "infer" ]]; then
    echo "Usage: bash slurm/submit.sh {server|train|infer} [extra sbatch args...]" >&2
    exit 1
fi
shift

CONFIG="${SUBMIT_CONFIG:-$SCRIPT_DIR/config.env}"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: $CONFIG not found. cp slurm/config.env.example slurm/config.env" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG"

PROJECT_DIR="${PROJECT_DIR:-$PROJECT_ROOT}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

if [[ -z "${SLURM_PARTITION:-}" ]]; then
    echo "ERROR: SLURM_PARTITION is empty in $CONFIG. Check partitions with: sinfo -s" >&2
    exit 1
fi

# SLURM refuses the job silently-ish if the log directory is missing (the job
# dies instantly with no output file to explain why), so create it here.
mkdir -p "$LOG_DIR"

ARGS=(
    --partition="$SLURM_PARTITION"
    --gres="gpu:${SLURM_GPUS:-1}"
    --cpus-per-task="${SLURM_CPUS:-8}"
    --mem="${SLURM_MEM:-64G}"
)
[[ -n "${SLURM_ACCOUNT:-}" ]] && ARGS+=(--account="$SLURM_ACCOUNT")

case "$ROLE" in
server)
    ARGS+=(
        --time="${SERVER_TIME:-08:00:00}"
        --output="$LOG_DIR/server_%j.out" --error="$LOG_DIR/server_%j.err"
    )
    SCRIPT="$SCRIPT_DIR/start_label_server.sbatch"
    ;;
train)
    ARGS+=(
        --time="${TRAIN_TIME:-24:00:00}"
        --output="$LOG_DIR/train_%j.out" --error="$LOG_DIR/train_%j.err"
    )
    SCRIPT="$SCRIPT_DIR/train_bundle.sbatch"
    ;;
infer)
    ARRAY_SIZE="${INFER_ARRAY_SIZE:-4}"
    ARRAY_SPEC="0-$((ARRAY_SIZE - 1))"
    [[ -n "${INFER_ARRAY_THROTTLE:-}" ]] && ARRAY_SPEC="$ARRAY_SPEC%$INFER_ARRAY_THROTTLE"
    ARGS+=(
        --time="${INFER_TIME:-08:00:00}"
        --array="$ARRAY_SPEC"
        --output="$LOG_DIR/infer_%A_%a.out" --error="$LOG_DIR/infer_%A_%a.err"
    )
    SCRIPT="$SCRIPT_DIR/infer_wsi_batch.sbatch"
    ;;
esac

if [[ "${SUBMIT_DRYRUN:-0}" == "1" ]]; then
    echo sbatch "${ARGS[@]}" "$@" "$SCRIPT"
    exit 0
fi

cd "$PROJECT_ROOT"
JOB_OUTPUT="$(sbatch "${ARGS[@]}" "$@" "$SCRIPT")"
echo "$JOB_OUTPUT"

JOBID="$(grep -oE '[0-9]+$' <<<"$JOB_OUTPUT" || true)"
case "$ROLE" in
server)
    cat <<EOF

Watch it start:   squeue -u \$USER
Connection info:  cat $LOG_DIR/server_${JOBID:-<jobid>}.out
                  (repeats until the job leaves the queue and prints its box)
Stop it:          scancel ${JOBID:-<jobid>}
EOF
    ;;
train)
    echo
    echo "Progress: tail -f $LOG_DIR/train_${JOBID:-<jobid>}.out"
    ;;
infer)
    echo
    echo "Progress: ls $LOG_DIR/infer_${JOBID:-<jobid>}_*.out"
    ;;
esac
