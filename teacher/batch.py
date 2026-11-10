"""Run teacher trajectories for many tasks, N samples each, with a small worker pool.

Each worker gets its own MCP port (base_port + slot). Results land in runs/teacher/<task>/s<k>/.
Skips (task, sample) pairs that already have verify.json unless --force.

  python teacher/batch.py --tasks all --samples 2 --workers 2
  python teacher/batch.py --tasks httpx_3672 requests_6592 --samples 1
  python teacher/batch.py --tasks-file lists/train_tasks.txt --repo rich --samples 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_task_ids(args: argparse.Namespace) -> list[str]:
    if args.tasks_file:
        ids = [l.strip() for l in Path(args.tasks_file).read_text().splitlines() if l.strip() and not l.startswith("#")]
    elif args.tasks == ["all"]:
        ids = [json.loads(l)["instance_id"] for l in (ROOT / "data/gemma4/tasks.jsonl").read_text().splitlines() if l.strip()]
    else:
        ids = list(args.tasks)
    if args.repo:
        ids = [t for t in ids if t.rsplit("_", 1)[0] == args.repo]
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["all"])
    ap.add_argument("--tasks-file", default=None)
    ap.add_argument("--repo", default=None, help="filter by repo short name (fastapi|rich|requests|httpx)")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--base-port", type=int, default=8765)
    ap.add_argument("--results", default=str(ROOT / "runs" / "teacher"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args, extra = ap.parse_known_args()  # unknown flags (e.g. --model, --max-turns) go to run_task.py

    task_ids = load_task_ids(args)
    jobs = [(t, s) for t in task_ids for s in range(args.samples)]
    if not args.force:
        jobs = [(t, s) for t, s in jobs if not (Path(args.results) / t / f"s{s}" / "verify.json").exists()]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[batch] {len(jobs)} jobs, {args.workers} workers")
    if not jobs:
        return 0

    # sync once up front; workers run with --no-sync
    subprocess.run([str(ROOT / "scripts" / "sync.sh")], check=True)

    slots = list(range(args.workers))
    slot_lock = threading.Lock()
    done = {"ok": 0, "fail": 0}

    def run(job: tuple[str, int]) -> None:
        task, sample = job
        with slot_lock:
            slot = slots.pop()
        try:
            port = args.base_port + slot
            cmd = [
                sys.executable, str(ROOT / "teacher" / "run_task.py"),
                "--task-id", task, "--sample", str(sample), "--port", str(port),
                "--results", args.results, "--no-sync", *extra,
            ]
            t0 = time.time()
            r = subprocess.run(cmd, text=True, capture_output=True)
            tail = (r.stdout.strip().splitlines() or [""])[-1]
            print(f"[batch] {task} s{sample} slot{slot} exit={r.returncode} {time.time() - t0:.0f}s :: {tail[:200]}")
            if r.returncode not in (0, 1):
                print(r.stderr[-1500:], file=sys.stderr)
            done["ok" if r.returncode == 0 else "fail"] += 1
        finally:
            with slot_lock:
                slots.append(slot)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run, jobs))
    print(f"[batch] resolved={done['ok']} unresolved/failed={done['fail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
