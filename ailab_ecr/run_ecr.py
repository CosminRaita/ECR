"""Launcher: runs an ECR script unmodified, then reports wall time and peak GPU memory.

ECR's train_rec.py prints neither, and both are what T1 needs to record. runpy executes
the real file with __name__ == "__main__", so ECR's own argument parsing and code path are
untouched -- this adds instrumentation without being a patch.

Usage, with the working directory set to src_emo:
    python run_ecr.py train_rec.py --dataset redial_gen ...
"""
import os
import runpy
import sys
import time

if len(sys.argv) < 2:
    sys.exit(__doc__)

script = sys.argv[1]
if not os.path.exists(script):
    sys.exit(f"no such script: {script} (run this with the working directory at src_emo)")

# ECR's modules import each other as top-level names, so src_emo must be importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
sys.argv = [script] + sys.argv[2:]

import torch  # noqa: E402  (after sys.path, before the run, so the report always works)

t0 = time.time()
try:
    runpy.run_path(script, run_name="__main__")
finally:
    dt = time.time() - t0
    print("\n" + "=" * 60, flush=True)
    print(f"wall time: {dt / 60:.1f} min ({dt:.0f} s)", flush=True)
    if torch.cuda.is_available():
        print(f"peak GPU memory allocated : {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
              flush=True)
        print(f"peak GPU memory reserved  : {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB",
              flush=True)
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("=" * 60, flush=True)
