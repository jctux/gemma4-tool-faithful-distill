"""Phase-2 verification of a patch for one task (runs on the sandbox host).

  python -m teacher.verify --task-id fastapi_11194 --data-dir /mnt/nvme/kaggle/gemma4 \
      --patch runs/teacher/fastapi_11194/patch.diff --out runs/teacher/fastapi_11194/verify.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teacher.harness_env import TaskEnv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--patch", required=True, help="path to unified diff (empty file → NO_PATCH)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--results-dir", default=None, help="scratch dir for harness artifacts")
    ap.add_argument("--image", default="swebench-sandbox:latest")
    args = ap.parse_args()

    patch = Path(args.patch).read_text(encoding="utf-8") if Path(args.patch).exists() else ""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = TaskEnv(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir or out.parent),
        instance_id=args.task_id,
        image=args.image,
    )
    if not patch.strip():
        result = {"instance_id": args.task_id, "repo": env.task.repo, "resolved": False, "error": "NO_PATCH"}
    else:
        result = env.verify(patch)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "test_output_tail"}))
    return 0 if result.get("resolved") else 1


if __name__ == "__main__":
    sys.exit(main())
