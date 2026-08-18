#!/usr/bin/env bash
# Set up the project's Python environment ON the cluster.
#
#     bash slurm/build_env.sh
#
# Run this on a login node (or an interactive node if your site forbids heavy
# work on login nodes -- the torch download is several GB).
#
# No container runtime, no root, no build privileges: this creates a plain
# virtualenv on shared storage and pip-installs everything into it.
#
#   1. python -m venv  ->  $VENV_DIR
#   2. pip install -r requirements.txt -r requirements-train.txt
#      (torch's PyPI wheel bundles its own CUDA libraries, so the GPU stack
#      needs no system CUDA install; the openslide-bin wheel carries
#      libopenslide, so no apt packages are needed either)
#   3. verify every import, so a broken environment fails here rather than
#      at 3am in a queued job
#
# The venv lives on shared storage ($VENV_DIR in config.env), so every compute
# node runs the exact same environment with no per-node setup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "$SCRIPT_DIR/config.env" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.env"
else
    echo "WARNING: slurm/config.env not found; using defaults."
    echo "         cp slurm/config.env.example slurm/config.env  and edit it."
fi

PROJECT_DIR="${PROJECT_DIR:-$PROJECT_ROOT}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -n "${PYTHON_MODULE:-}" ]]; then
    echo "Loading module: $PYTHON_MODULE"
    module load "$PYTHON_MODULE" 2>/dev/null || \
        echo "  (module load failed -- continuing, $PYTHON_BIN may already be on PATH)"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: '$PYTHON_BIN' is not on PATH.

  - Find a python module:   module avail 2>&1 | grep -i python
  - Then set PYTHON_MODULE (and PYTHON_BIN if the binary has another name)
    in slurm/config.env.
EOF
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "ERROR: $PYTHON_BIN is $("$PYTHON_BIN" --version 2>&1) -- this project needs Python 3.9+." >&2
    echo "       Load a newer interpreter via PYTHON_MODULE in slurm/config.env." >&2
    exit 1
fi

# The torch CUDA wheel alone is ~2.5 GB; keep pip's download cache on project
# storage, not a small $HOME or a shared /tmp.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_ROOT/.pip_cache}"
mkdir -p "$PIP_CACHE_DIR"

# Outbound proxy for PyPI, if the site needs one (set in config.env).
if [[ -n "${HTTPS_PROXY:-}" ]]; then
    export HTTPS_PROXY https_proxy="$HTTPS_PROXY"
fi
if [[ -n "${HTTP_PROXY:-}" ]]; then
    export HTTP_PROXY http_proxy="$HTTP_PROXY"
fi

echo "Using:  $("$PYTHON_BIN" --version 2>&1) ($(command -v "$PYTHON_BIN"))"
echo "Venv:   $VENV_DIR"
echo "Cache:  $PIP_CACHE_DIR"
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
    # Import check with the venv's own interpreter, so a broken environment
    # fails here rather than in a queued job.
    "$VENV_DIR/bin/python" - <<'PYCHECK'
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

echo "== Step 1/3: virtualenv at $VENV_DIR =="
VENV_PARENT="$(dirname "$VENV_DIR")"
mkdir -p "$VENV_PARENT"
if confirm_replace "$VENV_DIR"; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo
echo "== Step 2/3: install project dependencies =="
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install \
    -r "$PROJECT_ROOT/requirements.txt" -r "$PROJECT_ROOT/requirements-train.txt"

echo
echo "== Step 3/3: verify imports =="
verify_env

cat <<EOF

==============================================================================
Python environment ready.

  Venv : $VENV_DIR

Next:
  1. Preflight on a GPU node (the gate for everything else):
         srun --partition=\${SLURM_PARTITION:-GPUA100} --gres=gpu:1 --pty \\
             bash slurm/check_env.sh
  2. Start the annotation server:
         bash slurm/submit.sh server
==============================================================================
EOF
