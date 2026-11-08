"""Submission invariants independent of GPU libraries."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path


def validate_submission(model: Path, workspace: Path):
    root = workspace.resolve()
    src = model.resolve()
    if not src.is_relative_to(root / "output") or not src.is_dir():
        raise ValueError("Submit a trained model directory under output/")
    device = root.stat().st_dev
    for f in src.rglob("*"):
        if not f.resolve().is_relative_to(root) or f.stat().st_dev != device:
            raise ValueError(
                "All submitted files must reside on the project filesystem"
            )
    return src


def prediction_counts(evaluation_dir: Path):
    dirs = sorted((evaluation_dir / "benchmark_results").glob("202*_*"), reverse=True)
    if not dirs:
        return []
    return [
        len(json.loads(p.read_text()))
        for p in (dirs[0] / "predictions").rglob("*.json")
    ]


def eligible(entry: dict, start_time: float, budget_seconds: int):
    if entry.get("error") or isinstance(entry.get("score"), bool):
        return False
    try:
        score = float(entry["score"])
        timestamp = datetime.fromisoformat(entry["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        elapsed = timestamp.timestamp() - start_time
    except (KeyError, ValueError, TypeError):
        return False
    return math.isfinite(score) and 0 <= score <= 100 and 0 <= elapsed <= budget_seconds
