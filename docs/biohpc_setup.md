# BioHPC setup

Everything in this project runs on the cluster. Nothing gets installed on your
own computer.

Defaults throughout are set for **UTSW BioHPC** (the Nucleus cluster,
GPUA100/GPUH100 partitions, `/project`–`/work`–`/archive` filesystems). On
another cluster the same steps apply — the site-specific values all live in one
config file, and anywhere marked `TODO` must be checked against your own
cluster's documentation or admins.

The environment is a plain Python virtualenv on shared storage: no container
runtime, no root, no system packages. Everything — CUDA-enabled PyTorch, MONAI,
MONAI Label, even the OpenSlide C library — installs from pip wheels.

The interactive annotation loop needs a live server plus a desktop session:
use the portal's **Web Visualization** facility (or BioHPC OnDemand) for the
desktop, as described below. Other BioHPC services that submit packaged batch
workflows through a web form are not part of this pipeline.

## 1. Get the code onto the cluster

```bash
ssh <you>@nucleus.biohpc.swmed.edu
cd /project/<your-group-or-user>        # NOT your home directory -- see below
git clone https://github.com/precal171/path_wsi_monai.git
cd path_wsi_monai
```

Put the checkout on a filesystem that is **large** and **visible to compute
nodes** — on UTSW BioHPC that means `/project` or `/work` space, not `$HOME`.
Two reasons: home quotas are small and the virtualenv alone is ~10 GB (torch's
CUDA wheel is most of it), and on some clusters home directories are not
mounted on compute nodes at all.

Useful reconnaissance while you are logged in:

```bash
sinfo -s                          # partitions (look for GPUA100 / GPUH100)
sacctmgr show assoc user=$USER    # accounts you may charge, if your site uses them
module avail 2>&1 | grep -i python
python3 --version                 # 3.9+ needed; if too old, note a module from above
```

## 2. Configure

```bash
cp slurm/config.env.example slurm/config.env
$EDITOR slurm/config.env
```

Fill in at minimum:

| Setting | What it is |
|---|---|
| `SLURM_PARTITION` | GPU queue — `GPUA100` or `GPUH100` on UTSW BioHPC |
| `SLURM_ACCOUNT` | allocation to charge, if your site uses one |
| `PYTHON_MODULE` | module providing Python 3.9+, if the default `python3` is too old |
| `PROJECT_DIR` | this checkout, on `/project` or `/work` |
| `WSI_DIR` | where your slides live — this is what QuPath will browse |
| `LOGIN_HOST` | login node for SSH tunnels (`nucleus.biohpc.swmed.edu`) |

`config.env` is gitignored, so your paths and account stay out of the repo.
Every script in `slurm/` reads this one file; nothing else needs editing.

## 3. Set up the Python environment

```bash
bash slurm/build_env.sh
```

This needs **no root and no special privileges** — it is a plain virtualenv
plus pip.

1. Creates a virtualenv at `VENV_DIR` **on shared storage**, so every compute
   node sees the exact same environment with no per-node setup.
2. Installs everything from `requirements.txt` + `requirements-train.txt`:
   torch (whose PyPI wheel bundles its own CUDA 12 libraries — no system CUDA
   needed), MONAI, MONAI Label, shapely, and the `openslide-bin` wheel that
   carries the OpenSlide C library, so no system packages are ever needed.
3. Verifies every import, including `monailabel`, before declaring success.

The download is several GB and takes a while the first time; pip's cache goes
to project storage (`PIP_CACHE_DIR`), not `$HOME` or `/tmp`. If your site needs
an outbound proxy for PyPI, set `HTTP_PROXY`/`HTTPS_PROXY` in `config.env`.

## 4. Preflight — the gate for everything else

Run the checker **on a GPU node** of the partition you configured:

```bash
srun --partition=GPUA100 --gres=gpu:1 --pty bash slurm/check_env.sh
```

Every line prints `PASS` or `FAIL`: the venv, **GPU visible to torch with a
supported compute capability**, `monailabel` and OpenSlide imports, project and
slide directories visible from the node, log directory writable. Do not
continue until everything passes — every confusing failure further down (silent
CPU-only inference, jobs dying with no output, "missing" data that is really a
missing mount) is one of these checks failing.

## 5. Start the annotation server

```bash
bash slurm/submit.sh server
squeue -u $USER
cat logs/server_<jobid>.out
```

`submit.sh` exists because `#SBATCH` directives are parsed before a script
runs, so the `.sbatch` files cannot read `config.env` themselves — submitting
them bare would request **no partition and no GPU**. The wrapper turns your
config into explicit `sbatch` flags. Never `sbatch slurm/*.sbatch` directly.

The job log prints the compute node, the port, and a ready-to-paste `ssh`
command. The port is randomised per job because compute nodes are shared and a
hard-coded 8000 eventually collides with someone else's job.

## 6. Connect QuPath

Two options. QuPath is a desktop GUI, so it has to run *somewhere* — but
neither option puts any Python, MONAI or CUDA on your machine.

### A. QuPath in a BioHPC Web Visualization session (preferred)

Start a session from the BioHPC portal's **Web Visualization** page (or BioHPC
OnDemand) — you get a GUI desktop running on the cluster, in your browser or a
VNC/DCV client. Run QuPath there and point it straight at the node:

```
http://<node>:<port>
```

Nothing is installed locally at all, and you are not moving slide pixels across
the internet, which makes panning around a slide far more responsive.

### B. QuPath on your own computer, over an SSH tunnel

Compute nodes accept no inbound connections from outside the cluster, so
forward a local port through the login node. On **your machine**:

```bash
LOGIN_HOST=nucleus.biohpc.swmed.edu bash slurm/tunnel.sh <node> <port>
```

Then point QuPath at `http://localhost:<port>`.

The node changes with every allocation, so rebuild the tunnel each session.
Leave the terminal open — closing it drops the tunnel (but not the server; use
`scancel` for that).

Extension install steps are in [../qupath/README.md](../qupath/README.md).

## Notes and gotchas

**The server holds a GPU for its whole walltime.** Request what an annotation
session actually needs. `scancel` it when you stop.

**Stay off GPUp4 with this stack.** Tesla P4 is compute capability sm_61;
recent PyTorch builds ship no kernels for it and fail with `no kernel image is
available for execution on the device`. `check_env.sh` flags this explicitly.

**Use OpenSlide, not cuCIM, to begin with.** cuCIM pins to a CUDA version and
is a common source of breakage on shared clusters where the host driver is
fixed and not yours to change. Get things working first; treat cuCIM as a later
optimisation.

**MONAI Label pulls in a fragile DICOM stack.** It imports its DICOM datastore
at module load even though a pathology workflow never touches DICOM, and recent
releases of its own dependencies conflict. `requirements-train.txt` pins the
combination that works and explains why. Symptom: the server never starts and
the traceback ends somewhere inside `pydicom` or `dicomweb_client`. Both
`build_env.sh` and `check_env.sh` verify the `monailabel` import so this
surfaces early.

**Compute nodes may not have internet.** The environment is fully materialised
on shared storage by step 3, so this only matters if you try to `pip install`
inside a job.

## Where things live on the cluster

```
$PROJECT_DIR/
├── venv/                         the whole environment (torch+CUDA, MONAI, MONAI Label)
├── .pip_cache/                   pip's wheel cache from build_env.sh
├── bundles/*/models/model.pt     trained weights (shared by server and batch jobs)
├── data/patches/                 extracted training patches + manifest.json
├── outputs/                      GeoJSON predictions
└── logs/                         SLURM job output
```

Because the server, training jobs and inference jobs all read and write the same
`models/` directory on cluster storage, a model trained by an overnight batch
run is live in QuPath as soon as you restart the server. Nothing to copy.
