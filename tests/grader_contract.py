"""CPU-only integration check against the actual pinned upstream Flask server.

Run inside the dependency image with /upstream and /harness mounted.
These adapter-only fixtures are rejected before GPU inference; no task score is claimed.
"""

import json
import runpy
import sys
import time
from pathlib import Path
from unittest.mock import patch

control = Path("/control")
control.mkdir(exist_ok=True)
(control / "meta").mkdir(exist_ok=True)
(control / "baseline.json").write_text(json.dumps({"score": 63.91}))
(control / "meta/run_meta.json").write_text(
    json.dumps({"start_time": int(time.time()), "timeout_s": 3600})
)
model = Path("/workspace/output/adapter")
model.mkdir(parents=True, exist_ok=True)
(model / "config.json").write_text("{}")
(model / "adapter_config.json").write_text("{}")
sys.argv = ["grader.py", "serve"]
with patch("flask.Flask.run"):
    namespace = runpy.run_module("aegisrl_bench.grader", run_name="__main__")
client = namespace["upstream"].app.test_client()
assert client.get("/health").status_code == 200
assert client.post("/set_baseline", json={"score": 100}).status_code == 403
assert namespace["server"].baseline_score == 63.91
assert client.post("/submit", json={"model_path": "../../control"}).status_code == 400
response = client.post("/submit", json={"model_path": "output/adapter"})
assert response.status_code == 200, response.get_data(as_text=True)
body = response.get_json()
assert "LoRA adapter detected" in body.get("error", ""), body
assert body["score"] == 0
assert (control / "evaluations/001/result.json").exists()
print(
    json.dumps(
        {
            "health": True,
            "baseline_protected": True,
            "path_escape_rejected": True,
            "adapter_only_rejected_by_upstream": True,
            "failed_evaluation_recorded": True,
        }
    )
)
