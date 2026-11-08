"""Pinned inputs and portable host paths."""

import os
import shutil
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent
UPSTREAM_URL = "https://github.com/microsoft/RD-Agent.git"
UPSTREAM_REVISION = "484776c211e4fbbeef03e0ec00d6bbee7362a4f4"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
IMAGE = os.environ.get("AEGIS_IMAGE", "aegisrl-bench:local")


def state():
    return (
        Path(os.environ.get("AEGIS_STATE", REPOSITORY / ".state"))
        .expanduser()
        .resolve()
    )


def codex_binary():
    return (
        Path(
            os.environ.get(
                "CODEX_BIN",
                shutil.which("codex") or str(Path.home() / ".local/bin/codex"),
            )
        )
        .expanduser()
        .resolve()
    )


def codex_auth():
    return (
        Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
        / "auth.json"
    )


def dependency_mount():
    # Optional local development reuse; the default image contains all dependencies.
    path = os.environ.get("AEGIS_VENV")
    return ["-v", f"{Path(path).resolve()}:/opt/fullbench:ro"] if path else []


DATASET_CANONICAL_SHA256 = {
    "train": "5208b0c5a0e535bc35cc7fcc6d244fd18d493664891610b8e54b60cf0d936175",
    "test": "ac313242fd25bcce0152c9ce0f1db012a3b52b37b1d70dc0c5650b2ea835f5fd",
}
