import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aegisrl_bench.config import state

ROOT = state()
results = json.loads((ROOT / "comparison.json").read_text())
fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
for result in results:
    run = ROOT / "runs" / result["run"]
    label = result["mode"]
    started = result["started_at"]
    rows = []
    for line in (run / "gpu.jsonl").read_text().splitlines():
        obj = json.loads(line)
        try:
            util, mem, power = map(float, obj["sample"].split(","))
        except ValueError:
            continue
        rows.append(((obj["time"] - started) / 60, util, mem))
    axes[0].plot(
        [x[0] for x in rows], [x[1] for x in rows], label=label, alpha=0.7, lw=0.8
    )
    scores = [
        s
        for s in result.get("eligible_submissions", result["submissions"])
        if not s.get("error")
    ]
    axes[1].plot(
        range(1, len(scores) + 1), [s["score"] for s in scores], marker="o", label=label
    )
    usage = result.get("usage_total", {})
    axes[2].bar(
        label,
        usage.get("uncached_input_tokens", 0),
        color="tab:blue",
        label="Uncached input" if label == results[0]["mode"] else None,
    )
    axes[2].bar(
        label,
        usage.get("output_tokens", 0),
        bottom=usage.get("uncached_input_tokens", 0),
        color="tab:orange",
        label="Output" if label == results[0]["mode"] else None,
    )
    if (run / "fault-events.jsonl").exists():
        full = None
        for line in (run / "fault-events.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["phase"] == "full":
                full = (e["time"] - started) / 60
            elif full is not None:
                axes[0].axvspan(
                    full, (e["time"] - started) / 60, color="red", alpha=0.055
                )
                full = None
axes[0].set(
    xlabel="Elapsed minutes",
    ylabel="GPU utilization (%)",
    title="Observed GPU trajectory (red shading: project filesystem full)",
)
axes[0].legend()
axes[1].axhline(
    results[0]["baseline_accuracy_percent"],
    color="gray",
    ls="--",
    lw=1,
    label="Base model",
)
axes[1].set(
    xlabel="Successful submission number",
    ylabel="GSM8K accuracy (%)",
    title="Full-test evaluation trajectory",
)
axes[1].legend()
axes[2].set(
    ylabel="Tokens",
    title="Codex usage (cached input is reported separately in comparison.json)",
)
axes[2].legend()
for ax in axes:
    ax.grid(alpha=0.2, axis="y")
fig.savefig(ROOT / "comparison.png", dpi=160)
fig.savefig(ROOT / "comparison.svg")

fig, axes = plt.subplots(
    len(results),
    1,
    figsize=(12, 4 * max(1, len(results))),
    squeeze=False,
    layout="constrained",
)
for row, result in enumerate(results):
    path = ROOT / "runs" / result["run"] / "training_trajectories.json"
    for trace in json.loads(path.read_text()):
        pairs = [
            (p.get("step", p.get("global_step")), p.get("loss", p.get("train_loss")))
            for p in trace["points"]
            if p.get("event") not in ["validation", "eval", "evaluation"]
        ]
        pairs = [
            (x, y)
            for x, y in pairs
            if isinstance(x, (int, float)) and isinstance(y, (int, float))
        ]
        if pairs:
            axes[row, 0].plot(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                label=trace["path"].removeprefix("artifacts/"),
                lw=0.8,
                alpha=0.7,
            )
    axes[row, 0].set(
        title=result["mode"] + " — agent-reported training loss",
        xlabel="Recorded optimizer step",
        ylabel="Loss",
    )
    axes[row, 0].legend(fontsize=7)
    axes[row, 0].grid(alpha=0.2)
fig.savefig(ROOT / "training_trajectories.png", dpi=160)
fig.savefig(ROOT / "training_trajectories.svg")
