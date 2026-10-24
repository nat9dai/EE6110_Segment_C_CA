"""Run N run_benchmark.py workers as separate Isaac Sim processes and restart any that crash.

    python3 scripts/launch_benchmark.py --workers 4 -- --exp main --n-scenes 60 --methods ...
"""
import argparse
import re
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.expanduser("~/isaacsim/python.sh")

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--max-restarts", type=int, default=6)
ap.add_argument("--stagger", type=float, default=20.0)
args, rest = ap.parse_known_args()
if rest and rest[0] == "--":
    rest = rest[1:]
exp = rest[rest.index("--exp") + 1] if "--exp" in rest else "main"
log_dir = os.path.join(ROOT, "results", "raw", exp, "logs")
os.makedirs(log_dir, exist_ok=True)

env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def start(w):
    logf = open(os.path.join(log_dir, f"w{w:02d}.log"), "a")
    cmd = [PY, "-u", os.path.join(ROOT, "scripts", "run_benchmark.py"), "--worker", str(w),
           "--n-workers", str(args.workers)] + rest
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT, env=env)


procs, restarts = {}, {w: 0 for w in range(args.workers)}
for w in range(args.workers):
    procs[w] = start(w)
    time.sleep(args.stagger)
while procs:
    time.sleep(10)
    for w, p in list(procs.items()):
        rc = p.poll()
        if rc is None:
            continue
        tail = open(os.path.join(log_dir, f"w{w:02d}.log")).read()[-4000:]
        if re.search(r"\] finished \d+ scenes", tail) and rc == 0:
            print(f"worker {w} finished", flush=True)
            del procs[w]
        elif restarts[w] < args.max_restarts:
            restarts[w] += 1
            print(f"worker {w} exited rc={rc}; restart {restarts[w]}", flush=True)
            procs[w] = start(w)
        else:
            print(f"worker {w} gave up after {restarts[w]} restarts", flush=True)
            del procs[w]
print("all workers done", flush=True)
