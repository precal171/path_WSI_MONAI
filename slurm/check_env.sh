#!/usr/bin/env bash
# Preflight: verify the whole stack on the node you are actually going to use.
#
# Run it ON A GPU NODE -- that is the point. From a login node:
#
#     srun --partition=GPUA100 --gres=gpu:1 --pty bash slurm/check_env.sh
#
# (use your SLURM_PARTITION). Every line prints PASS or FAIL; run the rest of
# the pipeline only when everything passes. Each check exists because its
# failure mode downstream is confusing -- a missing bind path aborts container
# startup, a CPU-only torch makes every QuPath inference take minutes, a
# Pascal-era GPU fails with "no kernel image is available", and a missing
# log directory kills sbatch jobs instantly with nothing written anywhere.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

FAILURES=0
pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; FAILURES=$((FAILURES + 1)); }

echo "== path_WSI_MONAI preflight on $(hostname -s) =="
echo

if [[ -f "$SCRIPT_DIR/config.env" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.env"
    pass "slurm/config.env found"
else
    fail "slurm/config.env missing -- cp slurm/config.env.example slurm/config.env"
    echo
    echo "Nothing else can be checked without it."
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$PROJECT_ROOT}"
CONTAINER="${CONTAINER:-$PROJECT_DIR/monai_base.sif}"
VENV_DIR="${VENV_DIR:-}"
APPTAINER_BIN="${APPTAINER_BIN:-singularity}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

if [[ -n "${APPTAINER_MODULE:-}" ]]; then
    if module load "$APPTAINER_MODULE" 2>/dev/null; then
        pass "module load $APPTAINER_MODULE"
    else
        fail "module load $APPTAINER_MODULE (check: module avail 2>&1 | grep -iE 'apptainer|singularity')"
    fi
fi

if command -v "$APPTAINER_BIN" >/dev/null 2>&1; then
    pass "$APPTAINER_BIN on PATH ($($APPTAINER_BIN --version 2>/dev/null))"
else
    fail "$APPTAINER_BIN not on PATH -- set APPTAINER_MODULE/APPTAINER_BIN in config.env"
fi

if [[ -f "$CONTAINER" ]]; then
    pass "container image at $CONTAINER"
else
    fail "no container at $CONTAINER -- run: bash slurm/build_container.sh"
fi

PY=python
BIND_ARGS=(--bind "$PROJECT_DIR:$PROJECT_DIR")
if [[ -n "$VENV_DIR" ]]; then
    if [[ -x "$VENV_DIR/bin/python" ]]; then
        pass "virtualenv at $VENV_DIR"
    else
        fail "no virtualenv at $VENV_DIR -- run: bash slurm/build_container.sh"
    fi
    PY="$VENV_DIR/bin/python"
    BIND_ARGS+=(--bind "$(dirname "$VENV_DIR"):$(dirname "$VENV_DIR")")
fi

# Bind paths that do not exist abort container startup with a cryptic error,
# so validate them here where the message can name the offender.
if [[ -n "${BIND_PATHS:-}" ]]; then
    IFS=',' read -ra PATHS <<<"$BIND_PATHS"
    for p in "${PATHS[@]}"; do
        if [[ -d "${p%%:*}" ]]; then
            pass "bind path $p exists"
        else
            fail "bind path $p does not exist on this node -- remove it from BIND_PATHS"
        fi
    done
    BIND_ARGS+=(--bind "$BIND_PATHS")
fi

run_in_env() {
    "$APPTAINER_BIN" exec --nv "${BIND_ARGS[@]}" "$CONTAINER" "$@" 2>/dev/null
}

if [[ -f "$CONTAINER" ]] && command -v "$APPTAINER_BIN" >/dev/null 2>&1; then
    GPU_REPORT="$(run_in_env "$PY" - <<'PYGPU'
import torch
if not torch.cuda.is_available():
    print("NOGPU")
else:
    cap = torch.cuda.get_device_capability(0)
    print(f"OK {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} torch {torch.__version__}")
PYGPU
)" || GPU_REPORT="ERROR"
    case "$GPU_REPORT" in
    OK*)
        pass "GPU visible in container: ${GPU_REPORT#OK }"
        SM="$(grep -oE 'sm_[0-9]+' <<<"$GPU_REPORT" | grep -oE '[0-9]+')"
        if [[ -n "$SM" && "$SM" -lt 70 ]]; then
            fail "compute capability sm_$SM is below what recent torch builds ship kernels for -- use a newer partition (GPUA100/GPUH100) or pin an older torch"
        fi
        ;;
    NOGPU)
        fail "torch.cuda.is_available() is False -- are you on a GPU node, with --nv working? (srun --gres=gpu:1 ... this script)"
        ;;
    *)
        fail "could not run torch inside the container"
        ;;
    esac

    if run_in_env "$PY" -c "import monailabel" >/dev/null; then
        pass "import monailabel"
    else
        fail "import monailabel -- rerun: bash slurm/build_container.sh (its pydicom stack is the usual culprit; see requirements-train.txt pins)"
    fi

    OPENSLIDE_VER="$(run_in_env "$PY" -c "import openslide; print(openslide.__library_version__)")" || OPENSLIDE_VER=""
    if [[ -n "$OPENSLIDE_VER" ]]; then
        pass "OpenSlide library $OPENSLIDE_VER"
    else
        fail "import openslide -- the openslide-bin wheel should provide the C library; rerun: bash slurm/build_container.sh"
    fi
fi

if [[ -d "${WSI_DIR:-/nonexistent}" ]]; then
    SLIDES=$(find "$WSI_DIR" -maxdepth 1 \( -iname '*.svs' -o -iname '*.ndpi' -o -iname '*.tif' -o -iname '*.tiff' \) 2>/dev/null | head -5 | wc -l)
    if [[ "$SLIDES" -gt 0 ]]; then
        pass "WSI_DIR $WSI_DIR readable, slides present"
    else
        fail "WSI_DIR $WSI_DIR exists but no slides found at its top level"
    fi
else
    fail "WSI_DIR '${WSI_DIR:-}' does not exist on this node -- compute nodes sometimes lack mounts login nodes have"
fi

if mkdir -p "$LOG_DIR" 2>/dev/null && [[ -w "$LOG_DIR" ]]; then
    pass "LOG_DIR $LOG_DIR writable"
else
    fail "LOG_DIR $LOG_DIR not writable -- sbatch jobs will die instantly with no output"
fi

echo
if [[ "$FAILURES" -eq 0 ]]; then
    echo "All checks passed. Start the server with: bash slurm/submit.sh server"
    exit 0
fi
echo "$FAILURES check(s) failed -- fix these before running anything else."
exit 1
