#!/usr/bin/env bash
# Build the SSH tunnel to a running MONAI Label server (QuPath connection
# pattern B -- QuPath on your own computer, server on a BioHPC GPU node).
#
# Run this ON YOUR OWN MACHINE, not on the cluster:
#
#     bash slurm/tunnel.sh <jobid>
#     bash slurm/tunnel.sh <node> <port>
#
# You do not need this if QuPath is running inside a BioHPC remote-desktop
# session (pattern A) -- there, QuPath reaches the node directly.
#
# Why a tunnel at all: compute nodes normally accept no inbound connections from
# outside the cluster. The tunnel forwards a local port through the login node,
# which can reach them. The node changes with every allocation, so this cannot
# be a fixed setting -- rerun it each session.

set -euo pipefail

LOGIN_HOST="${LOGIN_HOST:-}"
SSH_USER="${SSH_USER:-$USER}"

usage() {
    cat <<EOF
Usage:
    bash slurm/tunnel.sh <jobid>          # look up the node from SLURM (run where squeue works)
    bash slurm/tunnel.sh <node> <port>    # values printed in logs/server_<jobid>.out

Environment:
    LOGIN_HOST   BioHPC login host to tunnel through (required)
    SSH_USER     your cluster username (default: \$USER = $USER)

Example:
    LOGIN_HOST=biohpc.university.edu bash slurm/tunnel.sh nucleus042 41337
EOF
    exit 1
}

[[ $# -ge 1 ]] || usage

if [[ -z "$LOGIN_HOST" ]]; then
    echo "ERROR: set LOGIN_HOST to your BioHPC login hostname." >&2
    echo "       e.g. LOGIN_HOST=biohpc.university.edu bash slurm/tunnel.sh $*" >&2
    exit 1
fi

if [[ $# -eq 1 ]]; then
    JOBID="$1"
    if ! command -v squeue >/dev/null 2>&1; then
        cat >&2 <<EOF
ERROR: squeue is not available here, so the node cannot be looked up from a job id.

Open logs/server_${JOBID}.out on the cluster -- it prints the node and port --
then rerun with both:

    bash slurm/tunnel.sh <node> <port>
EOF
        exit 1
    fi
    NODE="$(squeue -j "$JOBID" -h -o '%N')"
    [[ -n "$NODE" ]] || { echo "ERROR: job $JOBID is not running (squeue returned no node)." >&2; exit 1; }
    echo "Job $JOBID is on node: $NODE"
    read -r -p "Port (from logs/server_${JOBID}.out): " PORT
else
    NODE="$1"
    PORT="$2"
fi

[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "ERROR: port must be a number, got '$PORT'." >&2; exit 1; }

cat <<EOF

Opening tunnel:  localhost:$PORT  ->  $NODE:$PORT  (via $LOGIN_HOST)

Point QuPath's MONAI Label server URL at:

    http://localhost:$PORT

Leave this terminal open for the whole annotation session. Ctrl-C closes the
tunnel (it does not stop the server -- use scancel for that).

EOF

exec ssh -N -L "${PORT}:${NODE}:${PORT}" "${SSH_USER}@${LOGIN_HOST}"
