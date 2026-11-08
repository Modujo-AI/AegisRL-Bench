"""Private read-only observation of training logs, including subsequently deleted attempts."""

import json
import subprocess
import time
from aegisrl_bench.config import state

ROOT = state()
script = """import json
from pathlib import Path
r=Path('/shared');result={}
for folder in ['logs','code','output']:
 p=r/folder
 if p.exists():
  for f in p.rglob('*'):
   if f.is_file() and not f.is_symlink() and f.suffix in ['.jsonl','.csv','.log'] and f.stat().st_size<8000000:
    try:result[str(f.relative_to(r))]=f.read_text(errors='replace')
    except OSError:pass
print(json.dumps(result))"""
known = {}
while True:
    active = 0
    done = 0
    for run in (ROOT / "runs").glob("*"):
        if not (run / "meta.json").exists() or (run / "excluded.json").exists():
            continue
        if (run / "result.json").exists():
            done += 1
            continue
        active += 1
        p = subprocess.run(
            ["docker", "exec", run.name + "-peer", "python", "-c", script],
            capture_output=True,
            text=True,
        )
        if p.returncode:
            continue
        try:
            files = json.loads(p.stdout)
        except ValueError:
            continue
        for rel, data in files.items():
            key = (run.name, rel)
            old = known.get(key, "")
            if data == old:
                continue
            dest = run / "observed-training" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(data)
            with (run / "observed-training-events.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "path": rel,
                            "operation": "append"
                            if data.startswith(old)
                            else "replace",
                            "text": data[len(old) :] if data.startswith(old) else data,
                        }
                    )
                    + "\n"
                )
            known[key] = data
    time.sleep(15)
