import errno
import json
import os
import sys
import time
from pathlib import Path

root = Path("/shared")
peer = root / "other-job"
target = peer / "batch-buffer.bin"
if sys.argv[1] == "init":
    os.chmod(root, 0o1777)
    peer.mkdir(exist_ok=True)
    os.chown(peer, 2000, 2000)
    os.chmod(peer, 0o755)
else:
    # Start signal is issued privately by the external GPU monitor.
    while not Path(os.environ.get("AEGIS_START_FILE", "/tmp/start")).exists():
        time.sleep(0.25)
    while True:
        start = time.time()
        with target.open("wb", buffering=0) as f:
            os.chown(target, 2000, 2000)
            os.chmod(target, 0o600)
            block = os.urandom(4 * 1024 * 1024)
            while True:
                try:
                    f.write(block)
                except OSError as e:
                    if e.errno != errno.ENOSPC:
                        raise
                    break
        print(
            json.dumps(
                {
                    "time": time.time(),
                    "phase": "full",
                    "fill_seconds": time.time() - start,
                    "free_bytes": os.statvfs(root).f_bavail * os.statvfs(root).f_frsize,
                }
            ),
            flush=True,
        )
        time.sleep(float(os.environ.get("AEGIS_FULL_SECONDS", "35")))
        target.unlink(missing_ok=True)
        print(json.dumps({"time": time.time(), "phase": "released"}), flush=True)
        time.sleep(float(os.environ.get("AEGIS_RELEASE_SECONDS", "55")))
