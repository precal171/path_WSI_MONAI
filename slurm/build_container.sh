#!/usr/bin/env bash
# Build the project container ON BioHPC.
#
#     bash slurm/build_container.sh
#
# Run this on a login node (or an interactive node if your site forbids builds
# on login nodes -- check local policy, image builds are CPU and I/O heavy).
#
# Clusters vary in whether unprivileged users may build images at all. This
# script tries the options in order of preference and tells you what to do if
# none work:
#
#   1. apptainer build --fakeroot   (usual case on modern Apptainer)
#   2. apptainer build              (works where user namespaces are enabled)
#   3. pull the base image only, then install deps into a virtualenv on shared
#      storage (no build privileges needed at all)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "$SCRIPT_DIR/config.env" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.env"
else
    echo "WARNING: slurm/config.env not found; using defaults."
    echo "         cp slurm/config.env.example slurm/config.env  and edit it."
    CONTAINER="$PROJECT_ROOT/path_wsi_monai.sif"
    APPTAINER_BIN=apptainer
    APPTAINER_MODULE=""
fi

CONTAINER="${CONTAINER:-$PROJECT_ROOT/path_wsi_monai.sif}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
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
  - Then set APPTAINER_MODULE (and APPTAINER_BIN if it is called 'singularity')
    in slurm/config.env.
EOF
    exit 1
fi

echo "Using:      $($APPTAINER_BIN --version)"
echo "Definition: $DEF_FILE"
echo "Output:     $CONTAINER"
echo

if [[ -e "$CONTAINER" ]]; then
    read -r -p "$CONTAINER already exists. Rebuild? [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { echo "Keeping the existing image."; exit 0; }
fi

# Builds need a lot of scratch space; /tmp is often small and shared on login
# nodes. Point the cache at project storage instead.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$PROJECT_ROOT/.apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$PROJECT_ROOT/.apptainer_tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
echo "Cache: $APPTAINER_CACHEDIR"
echo

cd "$PROJECT_ROOT"

echo "== Attempt 1: build --fakeroot =="
if "$APPTAINER_BIN" build --fakeroot "$CONTAINER" "$DEF_FILE"; then
    echo
    echo "Built $CONTAINER"
    exit 0
fi

echo
echo "== Attempt 2: build without --fakeroot =="
if "$APPTAINER_BIN" build "$CONTAINER" "$DEF_FILE"; then
    echo
    echo "Built $CONTAINER"
    exit 0
fi

cat >&2 <<EOF

==============================================================================
Both build attempts failed. This usually means your site does not permit
unprivileged image builds.

Three ways forward, in order of preference:

1. Ask the BioHPC admins to build it for you, or to enable --fakeroot. Send
   them slurm/apptainer.def -- it is short and does nothing unusual.

2. Build it somewhere you do have privileges and copy the .sif over. The image
   is a single file:
       apptainer build path_wsi_monai.sif slurm/apptainer.def
       rsync -avP path_wsi_monai.sif <you>@<biohpc>:$PROJECT_ROOT/

3. Skip containers entirely and use a conda environment on the cluster:
       conda env create -f environment.yml
       conda activate path-wsi-monai
       pip install -e .
   Then set USE_CONDA=1 in slurm/config.env; the sbatch scripts will activate
   the environment instead of using apptainer.
==============================================================================
EOF
exit 1
