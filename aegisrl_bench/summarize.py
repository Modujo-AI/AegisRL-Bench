"""Extract real Codex usage and immutable evaluation evidence; no estimated dollar billing."""

import json
import re
from datetime import datetime, timezone
from aegisrl_bench.config import state
from aegisrl_bench.validation import eligible

ROOT = state()
all_results = []
for run in sorted((ROOT / "runs").glob("*")):
    if not (run / "result.json").exists():
        continue
    result = json.loads((run / "result.json").read_text())
    transcript = []
    usage = {}
    commands = []
    for line in (run / "agent.jsonl").read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "turn.completed":
            for k, v in event.get("usage", {}).items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
        item = event.get("item", {})
        if event.get("type") != "item.completed":
            continue
        if item.get("type") == "agent_message":
            transcript.append("AGENT\n" + item.get("text", ""))
        elif item.get("type") == "command_execution":
            commands.append(item)
            transcript.append(
                "COMMAND\n"
                + item.get("command", "")
                + "\nEXIT "
                + str(item.get("exit_code"))
                + "\n"
                + item.get("aggregated_output", "")
            )
    (run / "agent_trace.txt").write_text("\n\n".join(transcript))
    usage["uncached_input_tokens"] = usage.get("input_tokens", 0) - usage.get(
        "cached_input_tokens", 0
    )
    result["usage_total"] = usage
    result["commands"] = len(commands)
    result["fault_evidence_in_commands"] = [
        i
        for i, c in enumerate(commands)
        if re.search(
            r"No space left on device|Errno 28|unexpected pos|PytorchStreamWriter failed",
            c.get("aggregated_output", ""),
        )
    ]
    result["wall_seconds"] = result["finished_at"] - result["started_at"]
    timing = json.loads((run / "control/meta/run_meta.json").read_text())
    agent_start = result.get("agent_started_at", timing["start_time"])
    deadline = agent_start + timing["timeout_s"]
    result["late_submissions"] = [
        s
        for s in result["submissions"]
        if datetime.fromisoformat(s["timestamp"])
        .replace(tzinfo=timezone.utc)
        .timestamp()
        > deadline
    ]
    result["eligible_submissions"] = [
        s
        for s in result["submissions"]
        if eligible(s, agent_start, timing["timeout_s"])
    ]
    valid = [s for s in result["eligible_submissions"] if not s.get("error")]
    result["best_score"] = max((s["score"] for s in valid), default=None)
    result["time_to_first_valid_submission_seconds"] = min(
        (
            datetime.fromisoformat(s["timestamp"])
            .replace(tzinfo=timezone.utc)
            .timestamp()
            - agent_start
            for s in valid
        ),
        default=None,
    )
    result["baseline_accuracy_percent"] = json.loads(
        (run / "control/baseline.json").read_text()
    )["score"]
    result["evaluation_evidence"] = []
    for ev in sorted((run / "control/evaluations").glob("*")):
        evidence = {"directory": str(ev), "predictions": []}
        for p in ev.glob("benchmark_results/*/predictions/**/*.json"):
            try:
                obj = json.loads(p.read_text())
            except ValueError:
                continue
            evidence["predictions"].append(
                {
                    "path": str(p),
                    "entries": len(obj) if isinstance(obj, (dict, list)) else None,
                }
            )
        result["evaluation_evidence"].append(evidence)
    trajectories = []
    for p in sorted((run / "artifacts").rglob("*.jsonl")):
        if p.is_symlink():
            continue
        points = []
        for line in p.read_text(errors="replace").splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict) and any(
                k in obj for k in ["loss", "train_loss", "reward"]
            ):
                points.append(obj)
        if points:
            trajectories.append({"path": str(p.relative_to(run)), "points": points})
    (run / "training_trajectories.json").write_text(json.dumps(trajectories, indent=2))
    result["training_trajectory_files"] = [
        {"path": t["path"], "points": len(t["points"])} for t in trajectories
    ]
    result["artifact_index"] = [
        {"path": str(p.relative_to(run)), "bytes": p.stat().st_size}
        for p in (run / "artifacts").rglob("*")
        if p.is_file() and not p.is_symlink()
    ]
    (run / "summary.json").write_text(json.dumps(result, indent=2))
    all_results.append(result)
(ROOT / "comparison.json").write_text(json.dumps(all_results, indent=2))
print(
    json.dumps(
        [
            {
                k: r.get(k)
                for k in [
                    "run",
                    "mode",
                    "wall_seconds",
                    "agent_seconds",
                    "commands",
                    "usage_total",
                    "best_score",
                    "submissions",
                    "fault_evidence_in_commands",
                    "evaluation_evidence",
                ]
            }
            for r in all_results
        ],
        indent=2,
    )
)
