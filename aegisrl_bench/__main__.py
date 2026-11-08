"""Host CLI. GPU dependencies live exclusively in Docker."""

import argparse
import fcntl
import hashlib
import json
import os
import runpy
import shutil
import signal
import subprocess
import sys
from .config import (
    PACKAGE,
    IMAGE,
    UPSTREAM_URL,
    UPSTREAM_REVISION,
    state,
    codex_binary,
    codex_auth,
    dependency_mount,
)


def command(args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def doctor():
    command(["docker", "info", "--format", "{{.ServerVersion}}"])
    command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv",
        ]
    )
    binary = codex_binary()
    for name in ["codex", "codex-code-mode-host", "codex-exec-server"]:
        if not (binary.parent / name).is_file():
            raise RuntimeError(
                f"Missing Codex companion binary: {binary.parent / name}"
            )
    command([binary, "--version"])
    if not codex_auth().is_file():
        raise RuntimeError("Codex authentication is missing. Run codex login.")
    command(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"])
    print(
        "Host tools, Codex credentials, and Docker image are present. No credentials printed."
    )


def prepare():
    root = state()
    root.mkdir(parents=True, exist_ok=True)
    up = root / "upstream"
    if not (up / ".git").exists():
        command(
            ["git", "clone", "--filter=blob:none", "--no-checkout", UPSTREAM_URL, up]
        )
        command(["git", "-C", up, "checkout", "--detach", UPSTREAM_REVISION])
    else:
        actual = subprocess.check_output(
            ["git", "-C", up, "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != UPSTREAM_REVISION:
            raise RuntimeError(
                f"Unexpected upstream revision: {actual}; use a clean state directory."
            )
    command(
        [
            "docker",
            "run",
            "--rm",
            *dependency_mount(),
            "-v",
            f"{root}:/state",
            "-v",
            f"{PACKAGE}:/harness/aegisrl_bench:ro",
            "-e",
            "PYTHONPATH=/harness",
            "-e",
            "HF_HOME=/state/cache/huggingface",
            IMAGE,
            "python",
            "-m",
            "aegisrl_bench.download",
        ]
    )
    protocol = json.loads((PACKAGE / "protocol.json").read_text())
    protocol["runtime_image_id"] = subprocess.check_output(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True
    ).strip()
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (root / "environment").mkdir(exist_ok=True)
    shutil.copyfile(
        PACKAGE / "docker/requirements.lock", root / "environment/requirements.lock"
    )
    files = [
        up / "rdagent/scenarios/rl/autorl_bench/core/opencompass.py",
        up / "rdagent/components/benchmark/configs/opencompass_template.yaml",
        up / "rdagent/components/benchmark/configs/models.yaml",
    ]
    (root / "source_manifest.json").write_text(
        json.dumps(
            {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
            indent=2,
        )
    )


def check_gpu_idle():
    values = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    if any(float(x) > 1024 for x in values.splitlines()):
        raise RuntimeError(
            "A GPU is already in use. Stop the unrelated GPU job before starting a benchmark."
        )


def require_prepared():
    for path in [
        "upstream/.git",
        "assets/model/config.json",
        "assets/train.jsonl",
        "private/data/gsm8k/test.jsonl",
        "protocol.json",
    ]:
        if not (state() / path).exists():
            raise RuntimeError(f"Missing {path}; run prepare first.")


def report():
    runpy.run_module("aegisrl_bench.summarize", run_name="__main__")
    if not json.loads((state() / "comparison.json").read_text()):
        raise RuntimeError("No completed runs to report.")
    command(
        [
            "docker",
            "run",
            "--rm",
            "--cpus",
            "2",
            *dependency_mount(),
            "-v",
            f"{state()}:/state",
            "-v",
            f"{PACKAGE}:/harness/aegisrl_bench:ro",
            "-e",
            "PYTHONPATH=/harness",
            "-e",
            "AEGIS_STATE=/state",
            IMAGE,
            "python",
            "-m",
            "aegisrl_bench.plot",
        ]
    )
    runpy.run_module("aegisrl_bench.report", run_name="__main__")
    print("Report:", state() / "report.html")


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(
        description="AegisRL-Bench: evaluating agentic RL under environment failures"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["build", "doctor", "prepare", "baseline", "report", "smoke"]:
        sub.add_parser(name)
    run = sub.add_parser("run")
    run.add_argument(
        "--condition", choices=["healthy", "enospc", "pair"], default="pair"
    )
    run.add_argument("--budget-seconds", type=int, default=3600)
    run.add_argument(
        "--agent-model", default=os.environ.get("AEGIS_AGENT_MODEL", "gpt-6-astra")
    )
    args = parser.parse_args()
    root = state()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "build":
        command(
            [
                "docker",
                "build",
                "-f",
                PACKAGE / "docker/Dockerfile",
                "-t",
                IMAGE,
                PACKAGE,
            ]
        )
        return
    if args.command == "doctor":
        doctor()
        return
    if args.command == "prepare":
        prepare()
        return
    if args.command == "report":
        report()
        return
    if args.command == "smoke":
        from .smoke import smoke

        smoke()
        return
    require_prepared()
    with (root / "experiment.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another experiment is already using this state directory."
            )
        check_gpu_idle()
        from . import runner

        if args.command == "baseline":
            runner.baseline()
            return
        if args.budget_seconds <= 0:
            parser.error("--budget-seconds must be positive")
        baseline = root / "private/baseline.json"
        if not baseline.exists() or json.loads(baseline.read_text()).get("error"):
            raise RuntimeError(
                "A successful full baseline is required. Run baseline first."
            )
        doctor()
        runner.BUDGET = args.budget_seconds
        runner.MODEL = args.agent_model
        protocol = json.loads((root / "protocol.json").read_text())
        protocol["time_budget_seconds_per_condition"] = args.budget_seconds
        protocol["agent"]["model"] = args.agent_model
        protocol["agent"]["client"] = subprocess.check_output(
            [codex_binary(), "--version"], text=True
        ).strip()
        protocol["baseline_accuracy_percent"] = json.loads(baseline.read_text())[
            "score"
        ]
        (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
        with (root / "observer.log").open("a") as log:
            observer = subprocess.Popen(
                [sys.executable, "-m", "aegisrl_bench.observe"],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                modes = (
                    ["healthy", "enospc"]
                    if args.condition == "pair"
                    else [args.condition]
                )
                for mode in modes:
                    runner.trial(mode)
            finally:
                observer.terminate()
                try:
                    observer.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    observer.kill()
                    observer.wait()
        report()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
