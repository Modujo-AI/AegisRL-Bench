"""Infrastructure wrapper around the pinned, unmodified upstream evaluator/server."""

import json
import shutil
import sys
import time
from pathlib import Path
from flask import request, jsonify
from aegisrl_bench.validation import validate_submission, prediction_counts
from rdagent.scenarios.rl.autorl_bench.benchmarks import get_evaluator
from rdagent.scenarios.rl.autorl_bench.core import server as upstream

CONTROL = Path("/control")
CONTROL.mkdir(exist_ok=True)
if not (CONTROL / "rdagent").exists():
    (CONTROL / "rdagent").symlink_to("/upstream/rdagent")
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
if sys.argv[1] == "baseline":
    result = get_evaluator("gsm8k").run_eval(
        "/model", str(CONTROL / "baseline"), model_name=BASE, gpu_count=1
    )
    if not result.get("error") and prediction_counts(CONTROL / "baseline") != [1319]:
        result = {
            **result,
            "score": 0.0,
            "error": "Baseline did not produce exactly 1319 predictions",
        }
    (CONTROL / "baseline.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    sys.exit(1 if result.get("error") else 0)


class IsolatedEvaluator:
    def run_eval(self, model_path, workspace_path, **kwargs):
        index = (
            len(list((CONTROL / "evaluations").glob("*")))
            if (CONTROL / "evaluations").exists()
            else 0
        )
        dest = CONTROL / "evaluations" / f"{index + 1:03d}"
        dest.mkdir(parents=True)
        started = time.time()
        result = {"score": 0.0, "error": "Evaluation did not complete"}
        try:
            src = validate_submission(Path(model_path), Path("/workspace"))
            shutil.copytree(src, dest / "model")
            result = get_evaluator("gsm8k").run_eval(
                str(dest / "model"), str(dest), **kwargs
            )
            counts = prediction_counts(dest)
            if not result.get("error") and counts != [1319]:
                result = {
                    **result,
                    "score": 0.0,
                    "error": f"Expected one complete 1319-example prediction file; observed {counts}",
                }
        except Exception as exc:
            result = {"score": 0.0, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            (dest / "result.json").write_text(
                json.dumps(
                    {"started": started, "finished": time.time(), **result}, indent=2
                )
            )
        return result


server = upstream.init_server("gsm8k", BASE, "/workspace")
server.scores_file = CONTROL / "scores.json"
server.get_evaluator = lambda: IsolatedEvaluator()
server.set_baseline(json.loads((CONTROL / "baseline.json").read_text())["score"])
# Timing/control records must remain writable when the project filesystem is full.
original_update = upstream.update_run_meta
original_read = upstream.read_run_meta
upstream.update_run_meta = lambda workspace, **kwargs: original_update(
    CONTROL / "meta", **kwargs
)
upstream.read_run_meta = lambda workspace: original_read(CONTROL / "meta")


@upstream.app.before_request
def restrict_control():
    if request.path == "/set_baseline":
        return jsonify(error="Not available to task agents"), 403


upstream.app.run(host="0.0.0.0", port=5000, threaded=True)
