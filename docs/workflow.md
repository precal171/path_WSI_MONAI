# The annotation → training → inference loop

Assumes [biohpc_setup.md](biohpc_setup.md) is done: environment built, server
running, QuPath connected.

The idea is that you never annotate a whole slide. You annotate a little, train,
let the model pre-annotate the next region, and correct it. Correcting is far
faster than drawing, so each round costs less than the one before.

## 1. Annotate

In QuPath, open a slide from your `WSI_DIR`.

Draw nuclei with the brush or wand tool and assign them the class **`Nucleus`** —
the class name has to match the label in
`monai_pathology/lib/configs/nuclei_segmentation.py`, or the annotation is
ignored during patch extraction.

> **Annotate completely inside a bounded region.** This is the single decision
> that most affects whether the model learns anything. If you outline twelve
> nuclei in a field containing forty, the other twenty-eight become background
> in the training mask, and you are actively teaching the model that nuclei are
> background.
>
> The reliable way: draw a rectangle, class it **`ROI`**, and annotate every
> nucleus inside it. Then pass `--roi-labels ROI` during data prep and only
> those fully-annotated areas are used. A handful of complete regions beats a
> whole slide of partial ones.

Submit the annotations to the server when you are done with a region.

## 2. Let active learning pick the next region

Use **Next Sample** in the QuPath extension. The server returns an unlabelled
slide to work on next.

Currently this is uniform random selection, which is deliberate: until the model
is good, its uncertainty estimates are noise, and "annotate where the model is
confused" performs *worse* than random. Uncertainty sampling is scaffolded in
`lib/activelearning/epistemic.py` for once you have a trained model worth
listening to.

## 3. Train

**Interactive** — the Train button in QuPath. Runs a short fine-tune from the
current weights on the same GPU node as the server. Good for tightening a model
after a few new regions; takes seconds to minutes.

**Batch** — once annotations have accumulated, do it properly:

```bash
python tools/prepare_training_data.py \
    --input-dir "$WSI_DIR" \
    --output-dir data/patches \
    --labels Nucleus \
    --roi-labels ROI \
    --patch-size 256

bash slurm/submit.sh train
tail -f logs/train_<jobid>.out
```

Both paths run the *same* `bundles/nuclei_segmentation/configs/train.json`. Only
the epoch count and batch size differ, so an interactive improvement is a real
improvement and not an artefact of a second, subtly different training path.
See [bundle_design.md](bundle_design.md).

Watch `val_mean_dice` in the log. Rough expectations for nuclei: below ~0.5 means
something is wrong (usually too few or partial annotations); 0.7–0.85 is a usable
working model; above ~0.9 on a *single* slide is likely optimistic, because
train/validation were split by patch rather than by slide. Add slides.

## 4. Apply

**Interactive** — Run Inference in QuPath segments the region on screen.

**Across the archive:**

```bash
bash slurm/submit.sh infer
```

Each array task takes every Nth slide, writing one GeoJSON per slide into
`outputs/`. Size the array with `INFER_ARRAY_SIZE` in `slurm/config.env`, and
cap concurrency with `INFER_ARRAY_THROTTLE` if your allocation is tight.

Empty tiles are skipped by a cheap low-resolution tissue check first — on a real
slide, which is mostly glass, that is the difference between minutes and hours.

## 5. Review, and go round again

In QuPath: **Objects → Import objects**, choose the slide's `.geojson`.

Now correct rather than draw: delete false positives, fix boundaries, add what
was missed. Submit the corrections as new annotations and return to step 3. This
is the whole point of the loop — annotation effort per round falls as the model
improves.

## Practical notes

**Nothing is ever copied between machines.** The server, training jobs and
inference jobs all read and write `bundles/*/models/model.pt` on cluster storage.
After a batch training job finishes, restart the server (`scancel`, then
`bash slurm/submit.sh server`) and QuPath is using the new weights.

**Objects split across tile boundaries** are merged automatically when tiles
overlap (`--overlap`, default 64px). The merge joins polygons that actually
overlap, which distinguishes the two halves of one nucleus from two nuclei that
merely touch — a heuristic that occasionally fuses two real objects in dense
tissue. `--no-merge` disables it.

**Slide-wide runs produce a lot of objects.** Tens of thousands of nuclei make
QuPath sluggish if imported as annotations, so batch inference writes
`detection` objects by default (lighter, not individually editable). Use
`--object-type annotation` when you intend to edit them.

**Tune the output** with `--min-area` (drops speckle) and `--simplify` (cuts
polygon vertices several-fold with no visible change). Both meaningfully reduce
GeoJSON size on a full slide.

## When it goes wrong

| Symptom | Usual cause |
|---|---|
| QuPath cannot reach the server | Tunnel dropped, or the job ended. Check `squeue -u $USER`. |
| Inference returns nothing | No trained weights yet, or predictions never cross the threshold — check `val_mean_dice`. |
| "No annotations matching \['Nucleus'\]" | The QuPath class name does not match; it is case-sensitive. |
| Model never improves | Almost always partial annotation. Use `--roi-labels`. |
| Everything is slow | The GPU is not visible. Verify `--nv` (see biohpc_setup.md step 3). |
| Predictions offset from the tissue | A coordinate bug — `pytest tests/test_geojson_conversion.py` and open an issue. |
