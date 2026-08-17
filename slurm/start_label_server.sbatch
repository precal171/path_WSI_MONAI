#!/usr/bin/env bash
#SBATCH --job-name=monailabel-server
#SBATCH --output=logs/server_%j.out
#SBATCH --error=logs/server_%j.err
#
# DO NOT sbatch this file directly: #SBATCH directives are parsed before the
# shell runs, so nothing here can read config.env -- a bare submission gets NO
# partition and NO GPU. Submit through the wrapper, which passes the resources
# from slurm/config.env as sbatch flags:
#
#     bash slurm/submit.sh server
#
# Runs the MONAI Label annotation server on a GPU node so QuPath can drive it.
# Then read logs/server_<jobid>.out: it prints the node, the port, and the exact
# ssh command to reach it. See docs/biohpc_setup.md.
#
# The server holds the GPU for its whole walltime, so request what you will
# actually use for an annotation session -- not 24h out of habit.

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
CONTAINER="${CONTAINER:-$PROJECT_DIR/monai_base.sif}"
VENV_DIR="${VENV_DIR:-}"
APPTAINER_BIN="${APPTAINER_BIN:-singularity}"
WSI_DIR="${WSI_DIR:?Set WSI_DIR in slurm/config.env}"

# With the pull+venv setup, project commands live in the venv, not the image.
MONAILABEL_CMD=monailabel
[[ -n "$VENV_DIR" ]] && MONAILABEL_CMD="$VENV_DIR/bin/monailabel"

[[ -n "${APPTAINER_MODULE:-}" ]] && { module load "$APPTAINER_MODULE" 2>/dev/null || true; }

# Pick a free high port. Compute nodes are shared, and a hard-coded 8000 will
# eventually collide with another user's job; that failure is confusing because
# the server appears to start and then nothing can reach it.
PORT="${MONAILABEL_PORT:-0}"
if [[ "$PORT" == "0" ]]; then
    PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
fi

NODE="$(hostname -f)"
LOGIN_HOST="${LOGIN_HOST:-nucleus.biohpc.swmed.edu}"

cat <<EOF

================================================================================
 MONAI Label server starting
================================================================================
 Job              : ${SLURM_JOB_ID:-interactive}
 Compute node     : $NODE
 Port             : $PORT
 Slides (studies) : $WSI_DIR
 Container        : $CONTAINER
 GPU              : ${CUDA_VISIBLE_DEVICES:-none allocated}

 CONNECT FROM QUPATH
 -------------------
 A) QuPath running inside a BioHPC remote-desktop session (nothing installed on
    your own machine):

        http://$NODE:$PORT

 B) QuPath on your own computer -- open an SSH tunnel first, then point QuPath
    at localhost. Run this on YOUR machine, in a terminal you leave open:

        ssh -N -L ${PORT}:${NODE}:${PORT} ${USER}@${LOGIN_HOST}

    then in QuPath use:

        http://localhost:$PORT

    (slurm/tunnel.sh builds this command for you.)

 Stop the server with:  scancel ${SLURM_JOB_ID:-<jobid>}
================================================================================

EOF

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-$PROJECT_DIR/logs}"

if [[ ! -d "$WSI_DIR" ]]; then
    echo "ERROR: WSI_DIR '$WSI_DIR' does not exist or is not visible from this node." >&2
    echo "       Compute nodes sometimes lack mounts that login nodes have -- check with your admins." >&2
    exit 1
fi

BIND_ARG=""
[[ -n "${BIND_PATHS:-}" ]] && BIND_ARG="--bind $BIND_PATHS"
# Plain string, not an array: empty arrays trip `set -u` on the bash 4.2 that
# RHEL-era cluster nodes still ship.
VENV_BIND=""
[[ -n "$VENV_DIR" ]] && VENV_BIND="--bind $(dirname "$VENV_DIR"):$(dirname "$VENV_DIR")"

if [[ "${USE_CONDA:-0}" == "1" ]]; then
    # Fallback path for clusters without Apptainer (see build_container.sh).
    echo "Using conda environment rather than a container."
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate path-wsi-monai
    exec monailabel start_server \
        --app "$PROJECT_DIR/monai_pathology" \
        --studies "$WSI_DIR" \
        --host 0.0.0.0 --port "$PORT" \
        --conf models nuclei_segmentation \
        --conf bundle_root "${BUNDLE:-$PROJECT_DIR/bundles/nuclei_segmentation}" \
        --conf wsi_backend "${WSI_BACKEND:-OpenSlide}"
fi

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container not found at $CONTAINER -- run: bash slurm/build_container.sh" >&2
    exit 1
fi

# --nv exposes the GPU; without it the server silently runs on CPU and every
# interactive inference in QuPath takes tens of seconds instead of under one.
exec "$APPTAINER_BIN" exec --nv $BIND_ARG $VENV_BIND \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --bind "$WSI_DIR:$WSI_DIR" \
    "$CONTAINER" \
    "$MONAILABEL_CMD" start_server \
        --app "$PROJECT_DIR/monai_pathology" \
        --studies "$WSI_DIR" \
        --host 0.0.0.0 --port "$PORT" \
        --conf models nuclei_segmentation \
        --conf bundle_root "${BUNDLE:-$PROJECT_DIR/bundles/nuclei_segmentation}" \
        --conf wsi_backend "${WSI_BACKEND:-OpenSlide}"
