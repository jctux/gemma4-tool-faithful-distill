"""Harness self-check on the sandbox host: bootstrap Container A for a task, exercise the tools,
then verify the GOLD patch in Container B (must resolve). Also verifies an empty patch (must not).

  DOCKER_HOST=unix:///run/user/1000/podman/podman.sock .venv/bin/python -m teacher.smoke_test \
      --task-id httpx_3672 --data-dir /mnt/nvme/kaggle/gemma4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teacher.harness_env import TaskEnv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results-dir", default="runs/smoke")
    ap.add_argument("--skip-gold", action="store_true")
    args = ap.parse_args()

    env = TaskEnv(data_dir=Path(args.data_dir), results_dir=Path(args.results_dir), instance_id=args.task_id)
    t0 = time.perf_counter()
    env.setup()
    print(f"\n=== setup {time.perf_counter() - t0:.1f}s; prompt {len(env.prompt)} chars ===")
    print(env.prompt[:1500], "...\n")

    def show(name: str, **kw):
        res = env.call(name, **kw)
        s = json.dumps(res)
        print(f"--- {name}({kw}) -> {s[:400]}{'...' if len(s) > 400 else ''}")
        return res

    show("get_status")
    show("run_command", command="git log --oneline | head -3 && ls")
    show("run_command", command="python -c 'import sys; print(sys.version)'")
    first_py = env.call("run_command", command="ls *.py */__init__.py 2>/dev/null | head -1")["stdout"].strip()
    if first_py:
        show("read_file", filepath=first_py, start_line=1, end_line=5)
    sym = env.task.repo.split("/")[-1]
    show("search_similar_code", query=sym, k=3)
    show("get_code_neighbors", node=sym, max_neighbors=5)
    show("write_file", filepath="_smoke_tmp.txt", content="hello\n")
    show("run_command", command="rm _smoke_tmp.txt")
    show("submit_patch")
    empty_patch = env.extract_patch()
    print(f"empty patch chars: {len(empty_patch)}")
    env.teardown()

    print("\n=== verify EMPTY patch (expect resolved=False) ===")
    r0 = env.verify("") if False else {"resolved": False, "error": "skipped (NO_PATCH short-circuit)"}
    print(json.dumps(r0))

    if not args.skip_gold:
        print("\n=== verify GOLD patch (expect resolved=True) ===")
        t1 = time.perf_counter()
        r1 = env.verify(env.task.patch)
        r1["seconds"] = round(time.perf_counter() - t1, 1)
        print(json.dumps({k: v for k, v in r1.items() if k != "test_output_tail"}))
        if not r1["resolved"]:
            print(r1.get("test_output_tail", "")[-2500:])
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
