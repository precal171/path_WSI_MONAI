"""Regressions for the BioHPC deployment layer.

A review of the first cluster-deployment commit found the layer was built on
wrong assumptions about the site: docs described an "astrocyte desktop" that
does not exist (Astrocyte is UTSW's Nextflow workflow platform, not a remote
desktop), every sbatch script had its GPU/partition directives commented out
and deferred to a submit wrapper that was never written, config.env.example
defined SLURM settings nothing read while omitting keys the scripts did read,
and the container path led with a privileged build most cluster users cannot
run. Each test here pins one of those classes of defect shut.
"""

from __future__ import annotations

import os
import re
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLURM_DIR = os.path.join(REPO_ROOT, "slurm")

TEXT_EXTENSIONS = (".py", ".md", ".sh", ".sbatch", ".def", ".txt", ".json", ".yml", ".toml", ".conf", ".example")
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".apptainer_cache", ".apptainer_tmp", "node_modules"}


def _tracked_text_files():
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(TEXT_EXTENSIONS) or "." not in name:
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    yield path


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# Astrocyte: removed on request. It is a Nextflow workflow-submission platform
# at UTSW, not a remote desktop or a container runtime -- the docs that said
# otherwise sent users hunting for a UI that does not exist.
# ---------------------------------------------------------------------------


def test_no_astrocyte_anywhere():
    offenders = []
    for path in _tracked_text_files():
        if os.path.basename(path) == os.path.basename(__file__):
            continue
        if "astrocyte" in _read(path).lower():
            offenders.append(os.path.relpath(path, REPO_ROOT))
    assert not offenders, f"astrocyte mentioned in: {offenders}"


# ---------------------------------------------------------------------------
# Every variable the slurm scripts read from config.env must exist in
# config.env.example. The first version of this layer had eight keys the
# scripts read but the example never defined (USE_CONDA, LOGIN_HOST, the
# INFER_* family...), plus a whole SLURM block the scripts never read.
# ---------------------------------------------------------------------------

CONFIG_SCRIPTS = [
    "start_label_server.sbatch",
    "train_bundle.sbatch",
    "infer_wsi_batch.sbatch",
    "build_container.sh",
    "submit.sh",
    "check_env.sh",
]

# Set by SLURM / the runtime / the script itself, not expected in config.env.
RUNTIME_VARS = {
    "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_COUNT",
    "SLURM_ARRAY_JOB_ID", "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES",
    "APPTAINER_CACHEDIR", "APPTAINER_TMPDIR",
    "SINGULARITY_CACHEDIR", "SINGULARITY_TMPDIR",
    "SUBMIT_CONFIG", "SUBMIT_DRYRUN", "OMP_NUM_THREADS",
    "JOBID",  # parsed out of sbatch's own output by submit.sh
}


def _example_keys():
    keys = set()
    for line in _read(os.path.join(SLURM_DIR, "config.env.example")).splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*)=", line)
        if match:
            keys.add(match.group(1))
    return keys


def test_config_example_covers_everything_the_scripts_read():
    example = _example_keys()
    missing = {}
    for script in CONFIG_SCRIPTS:
        content = _read(os.path.join(SLURM_DIR, script))
        read_vars = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*):[-?]", content))
        unknown = read_vars - example - RUNTIME_VARS
        if unknown:
            missing[script] = sorted(unknown)
    assert not missing, (
        f"scripts read config variables that config.env.example never defines: {missing}. "
        "Either add them to the example (users cannot set what they cannot see) or, "
        "for runtime-provided variables, extend RUNTIME_VARS here."
    )


def test_config_example_sources_cleanly_and_in_order():
    """`set -u` catches a variable used before it is defined (e.g. CONTAINER
    referencing $PROJECT_DIR above PROJECT_DIR's own assignment)."""
    result = subprocess.run(
        ["bash", "-c", "set -u; USER=testuser; unset PROJECT_DIR 2>/dev/null; "
         f"source '{os.path.join(SLURM_DIR, 'config.env.example')}'"],
        capture_output=True, text=True, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    assert result.returncode == 0, f"config.env.example failed to source:\n{result.stderr}"


# ---------------------------------------------------------------------------
# The submit wrapper: it must exist (the first version was referenced in
# comments but never written), and it must actually pass a partition and a GPU
# -- the whole reason it exists.
# ---------------------------------------------------------------------------


def _dryrun_submit(role, tmp_path):
    config = tmp_path / "config.env"
    config.write_text(
        "SLURM_PARTITION=GPUA100\n"
        "SLURM_ACCOUNT=testacct\n"
        "SLURM_GPUS=1\n"
        f"LOG_DIR={tmp_path / 'logs'}\n"
        "INFER_ARRAY_SIZE=6\n"
        "INFER_ARRAY_THROTTLE=2\n"
    )
    env = dict(os.environ, SUBMIT_CONFIG=str(config), SUBMIT_DRYRUN="1")
    result = subprocess.run(
        ["bash", os.path.join(SLURM_DIR, "submit.sh"), role],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, f"submit.sh {role} dry-run failed:\n{result.stderr}"
    return result.stdout


@pytest.mark.parametrize("role", ["server", "train", "infer"])
def test_submit_wrapper_requests_partition_and_gpu(role, tmp_path):
    command = _dryrun_submit(role, tmp_path)
    assert "--partition=GPUA100" in command
    assert "--gres=gpu:1" in command
    assert "--account=testacct" in command
    assert f"{role}" in command


def test_submit_wrapper_sizes_the_infer_array_from_config(tmp_path):
    """The array size was hard-coded #SBATCH --array=0-3, silently fixing the
    shard count at 4 regardless of configuration."""
    command = _dryrun_submit("infer", tmp_path)
    assert "--array=0-5%2" in command


def test_submit_wrapper_refuses_an_empty_partition(tmp_path):
    config = tmp_path / "config.env"
    config.write_text("SLURM_PARTITION=\n")
    env = dict(os.environ, SUBMIT_CONFIG=str(config), SUBMIT_DRYRUN="1")
    result = subprocess.run(
        ["bash", os.path.join(SLURM_DIR, "submit.sh"), "server"],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode != 0
    assert "SLURM_PARTITION" in result.stderr


def test_referenced_helper_scripts_exist():
    """start_label_server.sbatch used to defer its resources to a
    submit_server.sh that was never written."""
    pattern = re.compile(r"\b(submit[a-z_]*\.sh|check_env\.sh|build_container\.sh|tunnel\.sh)\b")
    missing = {}
    search_files = [os.path.join(SLURM_DIR, f) for f in os.listdir(SLURM_DIR)]
    for doc_dir in (os.path.join(REPO_ROOT, "docs"), REPO_ROOT):
        for name in os.listdir(doc_dir):
            if name.endswith(".md"):
                search_files.append(os.path.join(doc_dir, name))
    for path in search_files:
        if not os.path.isfile(path):
            continue
        for ref in set(pattern.findall(_read(path))):
            if not os.path.exists(os.path.join(SLURM_DIR, ref)):
                missing.setdefault(os.path.relpath(path, REPO_ROOT), []).append(ref)
    assert not missing, f"references to slurm helper scripts that do not exist: {missing}"


# ---------------------------------------------------------------------------
# Documented commands must not fall into the no-GPU trap: a bare
# `sbatch slurm/foo.sbatch` requests no partition and no GPU because every
# resource directive lives in submit.sh, not the .sbatch files.
# ---------------------------------------------------------------------------


def test_docs_never_tell_the_user_to_sbatch_directly():
    offenders = []
    doc_files = []
    for doc_dir in (os.path.join(REPO_ROOT, "docs"), REPO_ROOT, os.path.join(REPO_ROOT, "qupath")):
        for name in os.listdir(doc_dir):
            if name.endswith(".md"):
                doc_files.append(os.path.join(doc_dir, name))
    for path in doc_files:
        in_code_block = False
        for lineno, line in enumerate(_read(path).splitlines(), 1):
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block and re.search(r"^\s*sbatch\s+(--\S+\s+)*slurm/", line):
                offenders.append(f"{os.path.relpath(path, REPO_ROOT)}:{lineno}")
    assert not offenders, (
        f"docs show bare `sbatch slurm/...` at {offenders} -- these submit with no "
        "partition and no GPU; route through `bash slurm/submit.sh <role>` instead."
    )


# ---------------------------------------------------------------------------
# Container definition and requirements
# ---------------------------------------------------------------------------


def test_base_image_is_pinned_everywhere():
    """A moving :latest made rebuilds non-reproducible and invited CUDA drift."""
    def_from = re.search(r"^From:\s*(\S+)", _read(os.path.join(SLURM_DIR, "apptainer.def")), re.M)
    assert def_from, "apptainer.def must have a From: line"
    image = def_from.group(1)
    assert ":" in image and not image.endswith(":latest"), f"apptainer.def From is not pinned: {image}"

    example = _read(os.path.join(SLURM_DIR, "config.env.example"))
    base = re.search(r"^BASE_IMAGE=(\S+)", example, re.M)
    assert base, "config.env.example must define BASE_IMAGE"
    assert not base.group(1).endswith(":latest"), f"BASE_IMAGE is not pinned: {base.group(1)}"


def test_container_build_checks_the_monailabel_import():
    """The old build check skipped monailabel -- the fragile import and the
    whole point of the image -- so builds passed and the server died later."""
    assert '"monailabel"' in _read(os.path.join(SLURM_DIR, "apptainer.def"))
    assert '"monailabel"' in _read(os.path.join(SLURM_DIR, "build_container.sh"))


def test_no_unconstrained_torch_reinstall_in_def():
    """`pip install --no-deps torch || true` was a no-op that claimed to
    protect the base image's CUDA torch while doing nothing of the sort."""
    content = _read(os.path.join(SLURM_DIR, "apptainer.def"))
    assert "pip install --no-cache-dir --no-deps torch" not in content
    assert "torch-constraint" in content, "the def should constrain pip to the base image's torch"


def test_requirements_carry_the_openslide_wheel():
    """The pull+venv path has no apt: libopenslide must come from the
    openslide-bin wheel or `import openslide` dies at runtime."""
    content = _read(os.path.join(REPO_ROOT, "requirements.txt"))
    assert re.search(r"^openslide-bin", content, re.M), "requirements.txt must include openslide-bin"
    match = re.search(r"^openslide-python>=([0-9.]+)", content, re.M)
    assert match and tuple(int(x) for x in match.group(1).split(".")[:2]) >= (1, 4), (
        "openslide-python must be >=1.4.0 to auto-detect the openslide-bin wheel"
    )


def test_both_cache_variable_families_are_exported():
    """Only APPTAINER_* was exported; a `singularity` binary ignored it and
    sent multi-GB pulls into a small shared /tmp."""
    content = _read(os.path.join(SLURM_DIR, "build_container.sh"))
    for var in ("APPTAINER_CACHEDIR", "APPTAINER_TMPDIR", "SINGULARITY_CACHEDIR", "SINGULARITY_TMPDIR"):
        assert f"export {var}=" in content, f"build_container.sh must export {var}"


def test_every_shell_script_parses():
    for name in os.listdir(SLURM_DIR):
        if name.endswith((".sh", ".sbatch")):
            result = subprocess.run(
                ["bash", "-n", os.path.join(SLURM_DIR, name)], capture_output=True, text=True
            )
            assert result.returncode == 0, f"{name} has a syntax error:\n{result.stderr}"
