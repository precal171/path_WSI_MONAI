# Using this on BioHPC — a guide for researchers

This guide is for annotating slides and training models day-to-day. It assumes
someone technical has already done the **one-time setup** (building the
container — see [biohpc_setup.md](biohpc_setup.md) if that hasn't happened
yet). You do not need to know how to program to follow this guide.

You'll be doing two things back and forth: drawing on slides in a program
called **QuPath**, and typing two or three commands into a **terminal** (a
text window where you type commands instead of clicking). The terminal part is
short and you'll be copying and pasting the same few commands each time — you
don't need to understand what they mean, just where to paste them.

## The idea, in plain terms

You draw circles around cells (nuclei) on a few slides in QuPath. The computer
learns from your drawings and gets better at finding those cells on its own.
Then it goes and finds them on all your other slides, and you review what it
found and fix anything wrong. Each round you do, it gets better and you have
less to fix.

All of the computing happens on your university's shared research computer
(**BioHPC**), not on your laptop. You're just remote-controlling it.

## What you'll need

- A BioHPC account and access to its **Web Visualization portal** — this is a
  website where you get a virtual desktop running on the cluster, so that
  programs like QuPath appear in your browser as if they were on your own
  computer. At UTSW this is the "Web Visualization" page on the BioHPC portal
  (portal.biohpc.swmed.edu), also reachable through BioHPC OnDemand. Ask the
  BioHPC helpdesk how to log in if you don't already use it.
- Your slide images already on the cluster's storage. Ask whoever set this up
  (or your BioHPC storage admin) where they are — you'll need that folder path
  once, to give to IT/the technical setup person, and it's already configured
  if setup is done.
- QuPath, either already installed on the visualization desktop or installable
  from [qupath.github.io](https://qupath.github.io) — ask your BioHPC
  helpdesk if you're not sure it's there.

## Every time you want to work

### Step 1 — Log into the BioHPC visualization portal

Open the web address your BioHPC gave you, log in, and start (or resume) a
desktop session. After a short wait you'll see a full desktop in your browser,
much like a normal computer desktop.

### Step 2 — Open a terminal and start the model server

On that virtual desktop, find and open a **Terminal** application (it may be
in an Applications menu, a taskbar icon, or a right-click menu — the exact
spot depends on your BioHPC's setup).

In the terminal, type this and press Enter (replace `path_wsi_monai` with
wherever the project folder actually is — ask your technical setup person if
unsure):

```
cd path_wsi_monai
bash slurm/submit.sh server
```

You'll see a short message with a number — that's a **job ID**. This just
means "the model server is queued to start on the cluster's computers." It
usually starts within a minute or two, sometimes longer if the cluster is
busy.

Check whether it's ready and get the connection details:

```
cat logs/server_<jobid>.out
```

(replace `<jobid>` with the number you were given). Once ready, this prints a
box of information — the important line looks like:

```
http://gpu-node-042:41337
```

**Copy that whole address.** You'll paste it into QuPath in a moment. It will
be different every time you start the server, so always re-check this rather
than reusing an old one.

If nothing prints yet, the job is still waiting in the queue — wait a minute
and run the same `cat logs/server_<jobid>.out` command again.

### Step 3 — Open QuPath and connect it

Open QuPath on the same virtual desktop (not on your own laptop).

The first time, you'll need a one-time add-on installed — see
[../qupath/README.md](../qupath/README.md), or ask your technical setup
person to have done this already.

In QuPath's preferences, find the **MONAI Label** settings and paste in the
address you copied in Step 2. QuPath should now be able to "see" the model
server.

### Step 4 — Open a slide and draw

Open one of your slide images in QuPath, the same way you normally would.

Use the drawing/brush tool to circle a handful of cells (nuclei). Give each
one the class name **`Nucleus`** exactly (this has to match, or the computer
won't recognize your drawings as training examples) — there's a classes panel
in QuPath where you set this before or after drawing.

**Important:** try to circle *every* cell inside the area you're working in,
not just some of them. If you draw a box around a patch of tissue and only
mark half the cells in it, the computer will wrongly learn that the
unmarked cells aren't cells at all. It's better to fully annotate a small
area than to partially annotate a big one.

When you're happy with a region, use the extension's **Submit** button to send
your drawings to the server.

### Step 5 — Ask the model to learn from your drawings

In the QuPath MONAI Label panel, click **Train**. This tells the model server
to update itself using everything you've submitted so far. It usually takes a
few seconds to a few minutes, depending on how much you've annotated.

You can keep annotating other slides while it trains, or wait — it won't stop
you from doing either.

### Step 6 — Let the model try the rest for you

Click **Run Inference** on a new region or slide. The model will draw its own
best guess of where the cells are. Review it: delete anything wrong, fix
boundaries that are off, add anything it missed. Submit those corrections the
same way as Step 4 — corrections count as new training examples too, so the
model keeps improving.

Repeat Steps 5 and 6 as you go. Each round should need less correcting than
the last.

### Step 7 — Applying the model to lots of slides at once

Once you're happy with how the model performs, you can have it run across your
entire slide collection automatically, rather than one at a time in QuPath.
This is done from the terminal (ask your technical setup person to run this
for you, or follow [../docs/workflow.md](../docs/workflow.md) if you're
comfortable with the terminal):

```
bash slurm/submit.sh infer
```

This produces a results file for each slide that you can then load into
QuPath (**Objects → Import objects**) to review, the same way as Step 6.

### Step 8 — When you're done for the day

You can just close the browser tab — nothing is lost, since everything lives
on the cluster. If you want to free up the computer resources for others,
your technical contact can stop the server job; it isn't something you need to
worry about.

## Things that commonly trip people up

**"QuPath says it can't connect to the server."**
The address from Step 2 changes every time the server restarts. Re-run
`cat logs/server_<jobid>.out` to get the current one, and make sure you're
pasting the whole `http://...` address including the port number after the
colon.

**"I don't see any model to run."**
Until the model has been trained at least once (Step 5), there's nothing for
it to use yet — this is expected the very first time.

**"My drawings don't seem to have taught it anything."**
The most common reason is partial annotation — see the note in Step 4. Try
fully circling every cell in a smaller area rather than some cells in a larger
one.

**"Nothing happens when I type the commands."**
Make sure you're typing into the Terminal window on the BioHPC visualization
desktop, not a terminal on your own laptop — the project files and the
cluster's computing power only exist on BioHPC's side.

**Something else is broken.**
This project's technical documentation is in the [docs/](.) folder — pass
along the exact error message to whoever set this up for you. They may also
want to look at [testing.md](testing.md) and [biohpc_setup.md](biohpc_setup.md).

## A few words you'll see used

| Word | What it means here |
|---|---|
| **Terminal** | A window where you type commands instead of clicking. |
| **Job** / **job ID** | A task you've asked the shared cluster computer to run; it gets a number so you can check on it. |
| **GPU** | A special kind of computer chip that makes the AI model fast. The cluster has these; your laptop doesn't need one. |
| **Server** | The program running on the cluster that QuPath talks to. You start it once per work session (Step 2). |
| **Training** | The process of the model learning from your drawings. |
| **Inference** | The model making its own guesses, using what it has learned. |

## Not for clinical use

This is research software to speed up annotation work. It is not a medical
device and has not been validated for diagnostic use.
