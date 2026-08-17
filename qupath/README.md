# Connecting QuPath

QuPath is the annotation front end. It talks to the MONAI Label server over
REST: you draw, it submits; you click Run Inference, the server segments the
region on screen and sends back GeoJSON.

## Install

1. **QuPath** — from [qupath.github.io](https://qupath.github.io). Either on your
   own computer, or inside a BioHPC remote-desktop session (see below).

2. **The MONAI Label extension** — a community extension,
   `qupath-extension-monailabel`, distributed as a `.jar`. Check its release
   notes for which QuPath versions it supports; the extension API changed at
   QuPath 0.4 and again at 0.5, so a mismatched pair fails to load with little
   explanation.

   Install: `Extensions → Manage extensions → Open extensions directory`, drop
   the `.jar` in, restart QuPath.

3. **Point it at the server.** The extension adds a MONAI Label preferences
   section. Set the server URL to whichever applies:

   | How you run QuPath | URL |
   |---|---|
   | Inside a BioHPC remote desktop | `http://<node>:<port>` |
   | On your own computer, via `slurm/tunnel.sh` | `http://localhost:<port>` |

   `logs/server_<jobid>.out` prints the node and port. The port is randomised
   per job, and the node changes with every allocation, so expect these to differ
   each session.

## Which way to run QuPath

**Inside a BioHPC remote desktop** if your site offers one (Open OnDemand, VNC, a
web visualization portal). Nothing is installed on your machine, and slide pixels
are not crossing the internet, so panning is far more responsive.

**On your own computer with an SSH tunnel** otherwise. Only QuPath and the `.jar`
are local — no Python, MONAI or CUDA. Compute nodes accept no inbound connections
from outside the cluster, hence the tunnel:

```bash
LOGIN_HOST=<biohpc-login-host> bash slurm/tunnel.sh <node> <port>
```

Leave that terminal open for the session.

## Annotating

Use the class name **`Nucleus`** — it must match the label in
`monai_pathology/lib/configs/nuclei_segmentation.py` exactly, including case, or
the annotation is skipped during patch extraction.

Annotate *completely* inside a region you mark with the class **`ROI`**, then use
`--roi-labels ROI` during data prep. Partial annotation teaches the model that
the nuclei you skipped are background, which is the most common reason a model
fails to improve. [../docs/workflow.md](../docs/workflow.md) covers this.

## Importing batch predictions

After `bash slurm/submit.sh infer`, each slide has a `.geojson` in `outputs/`.

**Objects → Import objects**, pick the file.

Or in the script editor (`Automate → Script editor`) for several slides:

```groovy
// Import predictions for the current image.
// Adjust the path to wherever your outputs/ directory is visible from.
def path = "/path/to/outputs/" + GeneralTools.stripExtension(
        getCurrentServer().getMetadata().getName()) + ".geojson"

def file = new File(path)
if (!file.exists()) {
    print "No predictions at " + path
    return
}
def objects = PathIO.readObjects(file)
addObjects(objects)
resolveHierarchy()
print "Imported " + objects.size() + " objects"
```

Batch inference writes **detection** objects by default. A slide-wide run can
produce tens of thousands of nuclei, and importing those as annotations makes
QuPath sluggish. Detections are lighter but not individually editable — rerun
with `--object-type annotation` when you intend to correct them and feed them
back as training data.

## If it does not work

**QuPath cannot reach the server.** Is the job still running (`squeue -u $USER`)?
Did the tunnel drop? Test independently of QuPath:

```bash
curl http://localhost:<port>/info/
```

That should return JSON describing the app and its models. If `curl` works and
QuPath does not, the problem is the extension configuration, not the network.

**The extension does not appear.** Almost always a QuPath/extension version
mismatch. Check `Extensions → Installed extensions`, and QuPath's log for a
load error.

**No models are offered.** The server only advertises models that have weights.
Until you have trained once, `bundles/nuclei_segmentation/models/model.pt` does
not exist and inference is correctly withheld. Train first.

**Inference returns nothing.** Either untrained weights, or predictions not
crossing the threshold — check `val_mean_dice` from your training run. See
[../docs/testing.md](../docs/testing.md).

**Annotations appear in the wrong place.** A coordinate bug. Run
`pytest tests/test_geojson_conversion.py`, note your QuPath and extension
versions, and open an issue — the request format may have changed. The adapter
to fix is `_region_from_request` in
`monai_pathology/lib/infers/nuclei_segmentation.py`.

## A caveat

The exact REST payloads the extension exchanges with MONAI Label have varied
between releases, and could not be verified against a live QuPath while this was
built. The geometry handling is well tested; the *wrapping* is the part most
likely to need adjusting. It is deliberately isolated in `_region_from_request`
and `__call__` so a protocol change is a small, local fix.
