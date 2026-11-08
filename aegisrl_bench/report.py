import html
import json
from aegisrl_bench.config import state

r = state()
results = json.loads((r / "comparison.json").read_text())
protocol = json.loads((r / "protocol.json").read_text())


def esc(x):
    return html.escape(str(x))


rows = []
evidence = []
for x in results:
    u = x.get("usage_total", {})
    run = x["run"]
    valid = [
        s for s in x.get("eligible_submissions", x["submissions"]) if not s.get("error")
    ]
    vals = [
        x["mode"],
        x.get("best_score"),
        len(valid),
        round(x["agent_seconds"], 1),
        round(x.get("time_to_first_valid_submission_seconds") or 0, 1),
        x["commands"],
        u.get("uncached_input_tokens"),
        u.get("cached_input_tokens"),
        u.get("output_tokens"),
    ]
    rows.append("<tr>" + "".join("<td>" + esc(v) + "</td>" for v in vals) + "</tr>")
    links = [
        ("Codex 完整轨迹", "agent_trace.txt"),
        ("Codex 原始 JSONL", "agent.jsonl"),
        ("训练轨迹", "training_trajectories.json"),
        ("运行结果与证据索引", "summary.json"),
        ("训练代码与产物", "artifacts/"),
        ("官方评测及模型快照", "control/evaluations/"),
    ]
    evidence.append(
        "<li>"
        + esc(x["mode"])
        + ": "
        + " · ".join(f'<a href="runs/{run}/{path}">{name}</a>' for name, path in links)
        + "</li>"
    )
page = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>AegisRL-Bench 环境异常对照实验</title><style>body{font:16px/1.6 system-ui;margin:40px auto;max-width:1200px;padding:0 24px;color:#17202a}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border:1px solid #d9e1e8;text-align:right}th{background:#eef3f7}img{max-width:100%}a{color:#155bb5}code{background:#f1f3f5;padding:2px 5px}.note{background:#fff5dc;padding:16px}h1{font-size:28px}</style><h1>AegisRL-Bench：GSM8K 环境异常对照实验</h1>"""
page += f"<p>基于 Microsoft RD-Agent <code>{protocol['commit']}</code>。基础模型 Qwen2.5-1.5B-Instruct；Codex gpt-6-astra；每组预算 3600 秒；完整 1,319 题评测。原始模型基线：<strong>{protocol['baseline_accuracy_percent']:.2f}%</strong>。</p>"
page += "<p>异常为共享项目文件系统真实 ENOSPC：另一租户写满 24GiB 文件系统，满盘保持 35 秒、释放 55 秒并循环。两组容量相同。控制器和私有评分数据未挂载到 agent。此轮不包含 checkpoint 坏块/损坏。</p>"
page += (
    "<table><tr>"
    + "".join(
        "<th>" + h + "</th>"
        for h in [
            "环境",
            "最好准确率 %",
            "有效提交数",
            "Agent 时间 s",
            "首次有效提交 s",
            "命令数",
            "未缓存输入 tokens",
            "缓存输入 tokens",
            "输出 tokens",
        ]
    )
    + "</tr>"
    + "".join(rows)
    + "</table>"
)
page += '<p class="note">一组一个独立 agent run，属于流程与恢复能力验证，不是环境异常对模型质量的因果估计。训练方法由 agent 按上游规则选择，SFT 也被上游允许。Token 用量来自 Codex 原始事件；未将其虚构为实际美元账单。3600 秒取上游 CLI 默认值，文档中的 12 小时默认值未使用。</p>'
page += '<h2>运行与评测轨迹</h2><img src="comparison.png" alt="GPU、评分与 token 用量"><h2>训练轨迹</h2><img src="training_trajectories.png" alt="逐步训练 loss"><p>不同 agent 可以选择不同方法、批次与步数；loss 曲线用于审计各自训练过程，不能直接当作相同训练方案的收敛对照。</p>'
page += (
    "<h2>原始证据</h2><ul>"
    + "".join(evidence)
    + '</ul><p><a href="protocol.json">实验协议</a> · <a href="comparison.json">机器可读结果</a> · <a href="environment/requirements.lock">依赖锁定</a> · <a href="source_manifest.json">源码与数据哈希</a></p>'
)
page += "<p>上游评分代码未修改。外围适配包括 Docker 隔离、现有 Codex 登录方式、受保护的评测日志和运行时间记录、提交模型快照，以及禁止 agent 修改 baseline。准备阶段无效启动保留在 runs/*/excluded.json 中，不计入正式结果。</p></html>"
(r / "report.html").write_text(page)
