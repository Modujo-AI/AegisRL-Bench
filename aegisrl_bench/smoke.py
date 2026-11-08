"""No GPU or Codex required: verify actual ENOSPC and tenant boundaries in 16 MiB."""

import json
import secrets
import subprocess
from .config import IMAGE, PACKAGE, state


def smoke():
    name = "aegis-smoke-" + secrets.token_hex(4)
    volume = name + "-work"

    def docker(*args, check=True):
        return subprocess.run(
            ["docker", *map(str, args)], check=check, capture_output=True, text=True
        )

    try:
        docker(
            "volume",
            "create",
            "--driver",
            "local",
            "--opt",
            "type=tmpfs",
            "--opt",
            "device=tmpfs",
            "--opt",
            "o=size=16m,mode=1777",
            volume,
        )
        docker(
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "none",
            "--memory",
            "128m",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "CHOWN",
            "--cap-add",
            "DAC_OVERRIDE",
            "--cap-add",
            "FOWNER",
            "-e",
            "AEGIS_FULL_SECONDS=3",
            "-e",
            "AEGIS_RELEASE_SECONDS=3",
            "-v",
            f"{volume}:/shared",
            "-v",
            f"{PACKAGE / 'pressure.py'}:/pressure.py:ro",
            IMAGE,
            "sleep",
            "infinity",
        )
        docker("exec", name, "python", "/pressure.py", "init")
        docker(
            "exec",
            "-d",
            name,
            "sh",
            "-c",
            "python -u /pressure.py cycle >/tmp/events.jsonl 2>/tmp/errors.log",
        )
        docker("exec", name, "touch", "/tmp/start")
        probe = """import errno,json,os,time
from pathlib import Path
root=Path('/shared');evidence={};deadline=time.time()+20
while time.time()<deadline:
 if os.statvfs(root).f_bavail==0:
  try:(root/'probe.bin').write_bytes(b'x'*1024)
  except OSError as e:
   if e.errno==errno.ENOSPC:evidence['real_enospc']=True;break
 time.sleep(.1)
try:(root/'other-job/batch-buffer.bin').read_bytes()
except PermissionError:evidence['peer_contents_hidden']=True
except FileNotFoundError:pass
while time.time()<deadline:
 try:
  (root/'probe.bin').write_bytes(b'ok');evidence['write_recovers']=True;break
 except OSError as e:
  if e.errno!=errno.ENOSPC:raise
 time.sleep(.1)
print(json.dumps(evidence));assert all(evidence.get(k) for k in ['real_enospc','peer_contents_hidden','write_recovers']),evidence
"""
        result = docker("exec", "--user", "1000:1000", name, "python", "-c", probe)
        evidence = json.loads(result.stdout)
        evidence["events"] = [
            json.loads(x)
            for x in docker(
                "exec", name, "cat", "/tmp/events.jsonl"
            ).stdout.splitlines()
        ]
        state().mkdir(parents=True, exist_ok=True)
        (state() / "smoke.json").write_text(json.dumps(evidence, indent=2))
        print(json.dumps(evidence, indent=2))
    finally:
        docker("rm", "-f", name, check=False)
        docker("volume", "rm", volume, check=False)
