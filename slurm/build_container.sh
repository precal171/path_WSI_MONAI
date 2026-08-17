#!/usr/bin/env bash
# Set up the project's container environment ON the cluster.
#
#     bash slurm/build_container.sh                # default: pull + venv
#     bash slurm/build_container.sh --build-def    # opt-in: build from .def
#
# Run this on a login node (or an interactive node if your site forbids heavy
# work on login nodes -- image pulls are I/O heavy).
#
# DEFAULT PATH -- pull + venv, no build privileges needed:
#
#   Regular cluster users have no root and often no --fakeroot, but pulling a
#   published Docker image and layering a Python virtualenv over it needs no
#   privileges at all. That is exactly what UTSW BioHPC's own Singularity guide
#   recommends ("you can use Apptainer to run Docker containers directly").
#
#   1. pull  $BASE_IMAGE           -> $CONTAINER          (~10+ GB, once)
#   2. venv --system-site-packages -> $VENV_DIR           (keeps the image's
#      CUDA-matched torch/MONAI; adds only this project's extra deps, incl.
#      monailabel and the openslide-bin wheel that carries libopenslide)
#   3. verify every import inside the container
#
# OPT-IN PATH (--build-def) -- a fully self-contained image from
# slurm/apptainer.def, for sites that permit --fakeroot or unprivileged builds.
# If you use it, set VENV_DIR= (empty) in config.env so the sbatch scripts run
# the image's own Python.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODE=pull
[[ "${1:-}" == "--build-def" ]] && MODE=build

if [[ -f "$SCRIPT_DIR/config.env" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.env"
else
    echo "WARNING: slurm/config.env not found; using defaults."
    echo "         cp slurm/config.env.example slurm/config.env  and edit it."
fi

PROJECT_DIR="${PROJECT_DIR:-$PROJECT_ROOT}"
CONTAINER="${CONTAINER:-$PROJECT_DIR/monai_base.sif}"
BASE_IMAGE="${BASE_IMAGE:-docker://projectmonai/monai:1.4.0}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
APPTAINER_BIN="${APPTAINER_BIN:-singularity}"
DEF_FILE="$SCRIPT_DIR/apptainer.def"

if [[ -n "${APPTAINER_MODULE:-}" ]]; then
    echo "Loading module: $APPTAINER_MODULE"
    module load "$APPTAINER_MODULE" 2>/dev/null || \
        echo "  (module load failed -- continuing, $APPTAINER_BIN may already be on PATH)"
fi

if ! command -v "$APPTAINER_BIN" >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: '$APPTAINER_BIN' is not on PATH.

  - Find the right module:   module avail 2>&1 | grep -i -E 'apptainer|singularity'
  - Then set APPTAINER_MODULE (and APPTAINER_BIN, which is 'singularity' on
    UTSW BioHPC) in slurm/config.env.
EOF
    exit 1
fi

# Pulls and builds need a lot of scratch space; /tmp is often small and shared
# on login nodes. Point caches at project storage. Both variable families are
# exported because Apptainer reads APPTAINER_* and Singularity reads
# SINGULARITY_* -- exporting only one silently sends the other into /tmp.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$PROJECT_ROOT/.apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$PROJECT_ROOT/.apptainer_tmp}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-$APPTAINER_TMPDIR}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# Outbound proxy for Docker Hub, if the site needs one (set in config.env).
if [[ -n "${HTTPS_PROXY:-}" ]]; then
    export HTTPS_PROXY https_proxy="$HTTPS_PROXY"
fi
if [[ -n "${HTTP_PROXY:-}" ]]; then
    export HTTP_PROXY http_proxy="$HTTP_PROXY"
fi

echo "Using:  $($APPTAINER_BIN --version)"
echo "Image:  $CONTAINER"
echo "Cache:  $APPTAINER_CACHEDIR"
echo

confirm_replace() {
    # Returns 0 to (re)create the target, 1 to keep it. Non-interactive runs
    # keep the existing file -- a queued job must never hang on a prompt.
    local target="$1"
    [[ -e "$target" ]] || return 0
    if [[ ! -t 0 ]]; then
        echo "$target already exists; keeping it (non-interactive run)."
        return 1
    fi
    local reply
    read -r -p "$target already exists. Recreate it? [y/N] " reply || reply=""
    if [[ "$reply" =~ ^[Yy]$ ]]; then
        rm -rf "$target"
        return 0
    fi
    echo "Keeping the existing $target."
    return 1
}

verify_env() {
    # Import check inside the container (and venv, when one is used), so a
    # broken environment fails here rather than at 3am in a queued job.
    local py="python"
    local bind_args=(--bind "$PROJECT_ROOT:$PROJECT_ROOT")
    if [[ -n "$VENV_DIR" ]]; then
        py="$VENV_DIR/bin/python"
        bind_args+=(--bind "$(dirname "$VENV_DIR"):$(dirname "$VENV_DIR")")
    fi
    "$APPTAINER_BIN" exec "${bind_args[@]}" "$CONTAINER" "$py" - <<'PYCHECK'
import importlib
required = [
    "monai", "torch", "numpy", "openslide", "shapely", "skimage",
    "cv2", "PIL", "tifffile", "fire", "monailabel",
]
missing = []
for module in required:
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{module}: {exc}")
if missing:
    raise SystemExit("Environment check failed:\n  " + "\n  ".join(missing))
import torch
print("Environment check passed. torch", torch.__version__,
      "| CUDA available:", torch.cuda.is_available(),
      "(False is expected on a login node without a GPU)")
PYCHECK
}

if [[ "$MODE" == "build" ]]; then
    echo "== Building self-contained image from $DEF_FILE =="
    if confirm_replace "$CONTAINER"; then
        if ! "$APPTAINER_BIN" build --fakeroot "$CONTAINER" "$DEF_FILE"; then
            echo
            echo "== --fakeroot failed; retrying plain build =="
            if ! "$APPTAINER_BIN" build "$CONTAINER" "$DEF_FILE"; then
                cat >&2 <<EOF

==============================================================================
Both build attempts failed: your site does not permit unprivileged builds.

Use the default no-privilege path instead -- it is what this script does
without --build-def:

    bash slurm/build_container.sh

It pulls the published base image and layers a virtualenv over it; no build
privileges are needed anywhere.
==============================================================================
EOF
                exit 1
            fi
        fi
    fi
    echo
    echo "== Verifying imports in the built image =="
    VENV_DIR="" verify_env
    echo
    echo "Built $CONTAINER. Set VENV_DIR= (empty) in slurm/config.env so the"
    echo "sbatch scripts use the image's own Python."
    exit 0
fi

# --- Default path: pull + venv ----------------------------------------------

echo "== Step 1/3: pull $BASE_IMAGE =="
if confirm_replace "$CONTAINER"; then
    "$APPTAINER_BIN" pull "$CONTAINER" "$BASE_IMAGE"
fi

echo
echo "== Step 2/3: virtualenv at $VENV_DIR =="
if [[ -z "$VENV_DIR" ]]; then
    echo "ERROR: VENV_DIR is empty in config.env. The pull path needs a venv;" >&2
    echo "       set VENV_DIR, or use --build-def for a self-contained image." >&2
    exit 1
fi
VENV_PARENT="$(dirname "$VENV_DIR")"
mkdir -p "$VENV_PARENT"
if confirm_replace "$VENV_DIR"; then
    # --system-site-packages keeps the base image's CUDA-matched torch/MONAI
    # visible; pip then adds only what the image lacks.
    "$APPTAINER_BIN" exec --bind "$VENV_PARENT:$VENV_PARENT" "$CONTAINER" \
        python -m venv --system-site-packages "$VENV_DIR"
fi
"$APPTAINER_BIN" exec \
    --bind "$VENV_PARENT:$VENV_PARENT" --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
    "$CONTAINER" \
    "$VENV_DIR/bin/pip" install --no-cache-dir \
    -r "$PROJECT_ROOT/requirements.txt" -r "$PROJECT_ROOT/requirements-train.txt"

echo
echo "== Step 3/3: verify imports =="
verify_env

cat <<EOF

==============================================================================
Container environment ready.

  Image : $CONTAINER
  Venv  : $VENV_DIR

Next:
  1. Preflight on a GPU node (the gate for everything else):
         srun --partition=\${SLURM_PARTITION:-GPUA100} --gres=gpu:1 --pty \\
             bash slurm/check_env.sh
  2. Start the annotation server:
         bash slurm/submit.sh server
==============================================================================
EOF
