"""Descriptive audit of published evidence; never infer recovery from GPU activity."""

import argparse
import json
import math
from pathlib import Path


def pressure_intervals(events, start, end):
    """Clip controller-confirmed full intervals to agent execution, preserving censoring.

    Unmatched releases, repeated full events and unknown phases fail closed. A final
    full event without release is right-censored; end-of-observation is not recovery.
    Fill time before the `full` event is deliberately not counted as full time.
    """
    if not all(math.isfinite(x) for x in (start, end)) or end <= start:
        raise ValueError("A finite, positive observation window is required")
    intervals = []
    active = None
    previous = -math.inf
    for event in events:
        t = event["time"]
        if (
            isinstance(t, bool)
            or not isinstance(t, (float, int))
            or not math.isfinite(t)
        ):
            raise ValueError("Event time must be finite")
        if t < previous:
            raise ValueError("Controller events must be chronologically ordered")
        previous = t
        if event["phase"] == "full":
            if active is not None:
                raise ValueError("Repeated full event without a release")
            active = t
        elif event["phase"] == "released":
            if active is None:
                raise ValueError("Release without a preceding full event")
            left, right = max(start, active), min(end, t)
            if right > left:
                intervals.append(
                    {
                        "start": left,
                        "end": right,
                        "left_censored": active < start,
                        "right_censored": t > end,
                    }
                )
            active = None
        else:
            raise ValueError("Unknown controller phase")
    if active is not None and active < end:
        intervals.append(
            {
                "start": max(start, active),
                "end": end,
                "left_censored": active < start,
                "right_censored": True,
            }
        )
    seconds = sum(x["end"] - x["start"] for x in intervals)
    return {
        "full_seconds": seconds,
        "full_fraction": seconds / (end - start),
        "intersecting_full_intervals": len(intervals),
        "right_censored_intervals": sum(x["right_censored"] for x in intervals),
        "intervals": intervals,
    }


def audit(reference):
    results = []
    for mode in ("healthy", "enospc"):
        directory = reference / mode
        original = json.loads((directory / "result.json").read_text())
        usage = original["usage_total"]
        row = {
            key: original[key]
            for key in (
                "mode",
                "agent_seconds",
                "best_score",
                "commands",
                "time_to_first_valid_submission_seconds",
            )
        }
        row["uncached_input_tokens"] = usage["uncached_input_tokens"]
        row["output_tokens"] = usage["output_tokens"]
        row["valid_submissions"] = len(original["eligible_submissions"])
        row["error_text_command_indices"] = original["fault_evidence_in_commands"]
        row["error_text_note"] = (
            "Text matches, not counts of failures or diagnosed incidents."
        )
        row["pressure"] = None
        if (directory / "fault-events.jsonl").exists() and (
            directory / "timing.json"
        ).exists():
            timing = json.loads((directory / "timing.json").read_text())
            events = [
                json.loads(line)
                for line in (directory / "fault-events.jsonl").read_text().splitlines()
            ]
            row["pressure"] = pressure_intervals(
                events, timing["start_time"], timing["end_time"]
            )
            row["pressure"]["timing_note"] = (
                "Reference run_meta timestamps have whole-second precision."
            )
        # Explicit unknowns avoid turning missing instrumentation into a success score.
        row["time_to_detect_seconds"] = None
        row["verified_checkpoint_recovery_seconds"] = None
        row["verified_retained_optimizer_updates"] = None
        row["unknown_reason"] = (
            "Reference has no independent checkpoint-resume verifier or "
            "timestamped diagnosis annotations. Agent-reported step logs "
            "and finite sampled weights do not prove exact state recovery."
        )
        results.append(row)
    healthy, faulted = results
    return {
        "kind": "descriptive_reference_audit",
        "conditions": results,
        "differences_enospc_minus_healthy": {
            "accuracy_percentage_points": round(
                faulted["best_score"] - healthy["best_score"], 2
            ),
            "first_valid_result_seconds": faulted[
                "time_to_first_valid_submission_seconds"
            ]
            - healthy["time_to_first_valid_submission_seconds"],
            "uncached_input_tokens": faulted["uncached_input_tokens"]
            - healthy["uncached_input_tokens"],
            "output_tokens": faulted["output_tokens"] - healthy["output_tokens"],
            "valid_submissions": faulted["valid_submissions"]
            - healthy["valid_submissions"],
        },
        "limitations": [
            "One independent run per condition; differences are not causal estimates.",
            "Filesystem full time is exposure, not measured training downtime.",
            "A released filesystem does not prove the application recovered.",
            "GPU utilization includes evaluation and does not prove optimizer progress.",
            "Repeated official-test feedback makes best score adaptively selected.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("examples/reference"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.dumps(audit(args.reference), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
