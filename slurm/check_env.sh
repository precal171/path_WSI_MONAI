#!/usr/bin/env bash
# Preflight: verify the whole stack on the node you are actually going to use.
#
# Run it ON A GPU NODE -- that is the point. From a login node:
#
#     srun --partition=GPUA100 --gres=gpu:1 --pty bash slurm/check_env.sh
#
# (use your SLURM_PARTITION). Every line prints PASS or FAIL; run the rest of
# the pipeline only when everything passes. Each check exists because its
# failure mode downstream is confusing -- a CPU-only torch makes every QuPath
# inference take minutes, a Pascal-era GPU fails with "no kernel image is
# available", a missing filesystem mount looks like missing data, and a missing
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
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

PY="$VENV_DIR/bin/python"
if [[ -x "$PY" ]]; then
    pass "virtualenv at $VENV_DIR ($("$PY" --version 2>&1))"
else
    fail "no virtualenv at $VENV_DIR -- run: bash slurm/build_env.sh"
    PY=""
fi

if [[ -n "$PY" ]]; then
    GPU_REPORT="$("$PY" - 2>/dev/null <<'PYGPU'
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
        pass "GPU visible to torch: ${GPU_REPORT#OK }"
        SM="$(grep -oE 'sm_[0-9]+' <<<"$GPU_REPORT" | grep -oE '[0-9]+')"
        if [[ -n "$SM" && "$SM" -lt 70 ]]; then
            fail "compute capability sm_$SM is below what recent torch builds ship kernels for -- use a newer partition (GPUA100/GPUH100) or pin an older torch"
        fi
        ;;
    NOGPU)
        fail "torch.cuda.is_available() is False -- are you on a GPU node? (srun --gres=gpu:1 ... this script). If yes, the installed torch may be a CPU-only build: rerun bash slurm/build_env.sh"
        ;;
    *)
        fail "could not import torch from the venv -- rerun: bash slurm/build_env.sh"
        ;;
    esac

    if "$PY" -c "import monailabel" >/dev/null 2>&1; then
        pass "import monailabel"
    else
        fail "import monailabel -- rerun: bash slurm/build_env.sh (its pydicom stack is the usual culprit; see requirements-train.txt pins)"
    fi

    OPENSLIDE_VER="$("$PY" -c "import openslide; print(openslide.__library_version__)" 2>/dev/null)" || OPENSLIDE_VER=""
    if [[ -n "$OPENSLIDE_VER" ]]; then
        pass "OpenSlide library $OPENSLIDE_VER"
    else
        fail "import openslide -- the openslide-bin wheel should provide the C library; rerun: bash slurm/build_env.sh"
    fi
fi

# Compute nodes sometimes lack filesystem mounts that login nodes have, which
# downstream looks like missing data. Check the project checkout itself first.
if [[ -d "$PROJECT_DIR" && -f "$PROJECT_DIR/slurm/submit.sh" ]]; then
    pass "PROJECT_DIR $PROJECT_DIR visible on this node"
else
    fail "PROJECT_DIR '$PROJECT_DIR' is not this checkout (or not mounted on this node)"
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
