"""Export compact reference evidence without credentials, test data or model weights."""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

SECRET = re.compile(
    rb"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


def checked_copy(src, dest):
    data = src.read_bytes()
    if SECRET.search(data):
        raise RuntimeError(f"Potential credential in {src.name}; export stopped.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def export(source, out):
    results = json.loads((source / "comparison.json").read_text())
    if {r["mode"] for r in results} != {"healthy", "enospc"}:
        raise RuntimeError("Both completed conditions are required.")
    out.mkdir(parents=True, exist_ok=True)
    published = []
    hashes = {}
    for result in results:
        run = source / "runs" / result["run"]
        dest = out / result["mode"]
        dest.mkdir(exist_ok=True)
        keys = [
            "run",
            "mode",
            "model",
            "budget_s",
            "agent_seconds",
            "agent_exit_code",
            "timed_out",
            "commands",
            "usage_total",
            "best_score",
            "baseline_accuracy_percent",
            "time_to_first_valid_submission_seconds",
            "submissions",
            "eligible_submissions",
            "late_submissions",
            "fault_evidence_in_commands",
            "training_trajectory_files",
        ]
        clean = {k: result.get(k) for k in keys}
        clean["prediction_counts"] = [
            p["entries"]
            for e in result["evaluation_evidence"]
            for p in e["predictions"]
        ]
        (dest / "result.json").write_text(json.dumps(clean, indent=2))
        published.append(clean)
        for name in [
            "agent.jsonl",
            "agent_trace.txt",
            "fault-events.jsonl",
            "fault-trigger.json",
            "gpu.jsonl",
            "model_audit.json",
        ]:
            if (run / name).exists():
                hashes[f"{result['mode']}/{name}"] = checked_copy(
                    run / name, dest / name
                )
        for name in ["summary.md", "best_model.json"]:
            if (run / "artifacts" / name).exists():
                checked_copy(run / "artifacts" / name, dest / name)
        for p in (run / "artifacts/code").rglob("*"):
            if p.is_file() and p.suffix in [".py", ".sh"]:
                checked_copy(p, dest / "code" / p.relative_to(run / "artifacts/code"))
        for p in (run / "artifacts/logs").rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            rel = p.relative_to(run / "artifacts/logs")
            target = dest / "logs" / rel
            if p.suffix == ".jsonl":
                data = p.read_bytes()
                if SECRET.search(data):
                    raise RuntimeError(f"Potential credential in log {rel}")
                target = target.with_suffix(target.suffix + ".gz")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(gzip.compress(data, mtime=0))
            elif p.suffix in [".json", ".py", ".sh"] and p.stat().st_size < 2_000_000:
                checked_copy(p, target)
        for p in (run / "harness").glob("*"):
            if p.is_file():
                checked_copy(p, dest / "provenance" / p.name)
        trajectories = json.loads((run / "training_trajectories.json").read_text())
        compact = []
        for trace in trajectories:
            points = []
            for row in trace["points"]:
                if not any(k in row for k in ["step", "global_step"]):
                    continue
                points.append(
                    {
                        k: v
                        for k, v in row.items()
                        if isinstance(v, (int, float, bool))
                        or k in ["event", "timestamp"]
                    }
                )
            if points:
                compact.append({"source": trace["path"], "points": points})
        (dest / "training_metrics.json").write_text(json.dumps(compact, indent=2))
    (out / "comparison.json").write_text(json.dumps(published, indent=2))
    (out / "evidence_hashes.json").write_text(json.dumps(hashes, indent=2))
    for name in ["protocol.json", "comparison.png", "training_trajectories.png"]:
        if (source / name).exists():
            checked_copy(source / name, out / name)
    text = """# Reference: GSM8K, healthy versus periodic ENOSPC

Two independent Codex runs, each with a 3,600-second budget, using the full upstream 1,319-question evaluation. Base-model accuracy: **63.91%**.

| Condition | Best accuracy | Valid submissions | Agent seconds | First valid result seconds | Commands | Uncached input tokens | Cached input tokens | Output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for r in published:
        u = r["usage_total"]
        valid = [s for s in r["eligible_submissions"] if not s.get("error")]
        text += f"| {r['mode']} | {r['best_score']:.2f}% | {len(valid)} | {r['agent_seconds']:.1f} | {r['time_to_first_valid_submission_seconds']:.1f} | {r['commands']} | {u['uncached_input_tokens']:,} | {u['cached_input_tokens']:,} | {u['output_tokens']:,} |\n"
    text += """
![Execution and scoring](comparison.png)

![Training loss](training_trajectories.png)

Each condition directory includes the complete Codex event stream and readable trajectory, agent-written training code and summary, compressed JSONL logs (including generated solutions), per-step metrics, GPU samples and submission records. The fault condition additionally includes private controller event timestamps. All successful evaluations must have 1,319 predictions; counts are recorded in result.json. Full model snapshots and test predictions remain in the local run archive and are regenerated by reproduction, rather than checked into Git.

These are single-run exploratory results. Agent strategies differ, so an accuracy difference is not evidence that faults improve or impair learning. Token counts are measured usage, not dollar billing.

The reference was executed with the pre-packaging harness preserved under each condition's `provenance/`. The portable runner retains the task/scorer and adds direct temporary credential transfer, explicit submission-completeness validation and cleanup handling. Original machine paths in provenance are historical, not instructions to use those paths. Run the root README commands to reproduce the workflow.

Preparation included an excluded launch where a login-shell PATH hid the installed Python environment; it was stopped before model training and is not included in these results. Official task timing begins after environment preparation. Archival/cleanup time is outside agent duration.
"""
    (out / "README.md").write_text(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("examples/reference"))
    a = p.parse_args()
    export(a.source, a.output)
