# BioHPC setup

Everything in this project runs on the cluster. Nothing gets installed on your
own computer.

The values below (partition, account, module names, filesystem paths) differ
between sites and cannot be guessed — anywhere you see `TODO(BioHPC)`, check
against your own cluster's documentation or ask its admins.

## 1. Get the code onto the cluster

```bash
ssh <you>@<biohpc-login-host>
git clone https://github.com/precal171/path_wsi_monai.git
cd path_wsi_monai
```

Put the checkout on a filesystem the **compute nodes** can see. Some clusters do
not mount home directories on compute nodes; if that is true of yours, use
scratch or project space instead. A job that cannot see its own code fails in a
confusing way — usually `No such file or directory` on a path that clearly
exists when you check it from the login node.

```bash
sinfo -s                          # partitions
sacctmgr show assoc user=$USER    # accounts you may charge
module avail 2>&1 | grep -iE 'apptainer|singularity'
```

## 2. Configure

```bash
cp slurm/config.env.example slurm/config.env
$EDITOR slurm/config.env
```

Fill in at minimum:

| Setting | What it is |
|---|---|
| `SLURM_PARTITION` | GPU queue name |
| `SLURM_ACCOUNT` | allocation to charge, if your site uses one |
| `APPTAINER_MODULE` | module providing `apptainer` (or `singularity`) |
| `PROJECT_DIR` | this checkout, on compute-node-visible storage |
| `WSI_DIR` | where your slides live — this is what QuPath will browse |
| `BIND_PATHS` | filesystems to mount into the container (`/archive`, `/scratch`, …) |

`config.env` is gitignored, so your paths and account stay out of the repo.

## 3. Build the container

```bash
bash slurm/build_container.sh
```

This produces `path_wsi_monai.sif` — CUDA, PyTorch, MONAI, MONAI Label,
OpenSlide and everything else, in one file. It takes 10–30 minutes.

If your site forbids unprivileged builds, the script explains three ways
forward (ask an admin, build elsewhere and copy the `.sif` over, or fall back to
a conda environment with `USE_CONDA=1`).

Check it works, on a **GPU node** — `--nv` is what exposes the GPU, and without
it everything silently falls back to CPU:

```bash
srun --partition=<PARTITION> --gres=gpu:1 --pty \
    apptainer exec --nv path_wsi_monai.sif \
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints `False`, stop and fix it here. Every performance problem
downstream traces back to this.

## 4. Start the annotation server

```bash
mkdir -p logs
sbatch slurm/start_label_server.sbatch
squeue -u $USER
cat logs/server_<jobid>.out
```

The log prints the compute node, the port, and a ready-to-paste `ssh` command.
The port is randomised per job because compute nodes are shared and a hard-coded
8000 eventually collides with someone else's job.

## 5. Connect QuPath

Two options. QuPath is a desktop GUI, so it has to run *somewhere* — but neither
option puts any Python, MONAI or CUDA on your machine.

### A. QuPath inside a BioHPC remote-desktop session (preferred)

If your BioHPC provides a remote desktop, Open OnDemand, or a web visualization
portal — many do — launch a session, run QuPath there, and point it straight at
the node:

```
http://<node>:<port>
```

Nothing is installed locally at all, and you are not moving slide pixels across
the internet, which makes panning around a slide far more responsive.

### B. QuPath on your own computer, over an SSH tunnel

Compute nodes normally accept no inbound connections from outside the cluster,
so forward a local port through the login node. On **your machine**:

```bash
LOGIN_HOST=<biohpc-login-host> bash slurm/tunnel.sh <node> <port>
```

Then point QuPath at `http://localhost:<port>`.

The node changes with every allocation, so rebuild the tunnel each session.
Leave the terminal open — closing it drops the tunnel (but not the server; use
`scancel` for that).

Extension install steps are in [../qupath/README.md](../qupath/README.md).

## Notes and gotchas

**The server holds a GPU for its whole walltime.** Request what an annotation
session actually needs. `scancel` it when you stop.

**Use OpenSlide, not cuCIM, to begin with.** cuCIM pins to a CUDA version and is
a common source of breakage inside Apptainer where the host driver is fixed and
not yours to change. Get things working first; treat cuCIM as a later
optimisation.

**MONAI Label pulls in a fragile DICOM stack.** It imports its DICOM datastore at
module load even though a pathology workflow never touches DICOM, and recent
releases of its own dependencies conflict. `requirements-train.txt` pins the
combination that works and explains why. Symptom: the server never starts and
the traceback ends somewhere inside `pydicom` or `dicomweb_client`.

**Compute nodes may not have internet.** The container bundles everything, so
this only matters if you try to `pip install` inside a job.

## Where things live on the cluster

```
$PROJECT_DIR/
├── path_wsi_monai.sif            the container
├── bundles/*/models/model.pt     trained weights (shared by server and batch jobs)
├── data/patches/                 extracted training patches + manifest.json
├── outputs/                      GeoJSON predictions
└── logs/                         SLURM job output
```

Because the server, training jobs and inference jobs all read and write the same
`models/` directory on cluster storage, a model trained by an overnight `sbatch`
run is live in QuPath as soon as you restart the server. Nothing to copy.
