import json
from pathlib import Path
import torch
from safetensors import safe_open

base = Path("/base")
models = Path("/models")
results = []


def tensor_path(root, key):
    index = root / "model.safetensors.index.json"
    if index.exists():
        return root / json.loads(index.read_text())["weight_map"][key]
    return root / "model.safetensors"


for model in sorted(models.glob("*/model")):
    audit = {
        "path": str(model),
        "sampled_parameters": [],
        "adapter_config_present": (model / "adapter_config.json").exists(),
    }
    for layer in [0, 10, 20]:
        for projection in ["q_proj", "v_proj"]:
            key = f"model.layers.{layer}.self_attn.{projection}.weight"
            with safe_open(
                str(tensor_path(base, key)), framework="pt", device="cpu"
            ) as f:
                a = f.get_tensor(key).float()
            with safe_open(
                str(tensor_path(model, key)), framework="pt", device="cpu"
            ) as f:
                b = f.get_tensor(key).float()
            delta = b - a
            audit["sampled_parameters"].append(
                {
                    "name": key,
                    "elements": a.numel(),
                    "changed_elements": int(torch.count_nonzero(delta)),
                    "rms_delta": float(delta.square().mean().sqrt()),
                    "all_finite": bool(torch.isfinite(b).all()),
                }
            )
    audit["sampled_weights_differ_from_base"] = any(
        x["changed_elements"] > 0 for x in audit["sampled_parameters"]
    )
    results.append(audit)
print(json.dumps(results, indent=2))
