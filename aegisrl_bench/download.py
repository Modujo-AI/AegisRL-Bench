"""Run inside the dependency image; test data stays in the private grader directory."""

import hashlib
import json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download
from aegisrl_bench.config import (
    BASE_MODEL,
    MODEL_REVISION,
    DATASET_REVISION,
    DATASET_CANONICAL_SHA256,
)

root = Path("/state")
assets = root / "assets"
assets.mkdir(parents=True, exist_ok=True)
snapshot_download(
    BASE_MODEL,
    revision=MODEL_REVISION,
    local_dir=str(assets / "model"),
    ignore_patterns=["*.onnx", "*.tflite", "*.gguf"],
)
ds = load_dataset("openai/gsm8k", "main", revision=DATASET_REVISION)
private = root / "private/data/gsm8k"
private.mkdir(parents=True, exist_ok=True)
manifest = {
    "base_model": BASE_MODEL,
    "model_revision": MODEL_REVISION,
    "dataset_revision": DATASET_REVISION,
    "splits": {},
}
for split, expected in [("train", 7473), ("test", 1319)]:
    rows = [{"question": r["question"], "answer": r["answer"]} for r in ds[split]]
    assert len(rows) == expected, (split, len(rows))
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (private / (split + ".jsonl")).write_text(text)
    if split == "train":
        (assets / "train.jsonl").write_text(text)
    canonical = "\n".join(
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for r in rows
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == DATASET_CANONICAL_SHA256[split], (
        "Dataset contents differ from the reference task"
    )
    manifest["splits"][split] = {
        "examples": len(rows),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
    }
(root / "input_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
