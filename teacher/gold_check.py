"""Verify the GOLD patch of every task whose snapshot is present → runs/gold/index.jsonl.

Tasks whose gold patch does not resolve are unverifiable in this environment and are excluded
from teacher sampling / SFT. Resumable: skips tasks already in the index unless --force.

  DOCKER_HOST=unix:///run/user/1000/podman/podman.sock .venv/bin/python -m teacher.gold_check \
      --data-dir /mnt/nvme/kaggle/gemma4 [--repo fastapi] [--workers 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from swegemma.models.task import load_tasks  # noqa: E402
from teacher.harness_env import TaskEnv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="runs/gold/index.jsonl")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--task-ids", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data = Path(args.data_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out.exists() and not args.force:
        for l in out.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                done[r["instance_id"]] = r

    tasks = load_tasks(data / "tasks.jsonl")
    todo = []
    for t in tasks:
        if args.task_ids and t.instance_id not in args.task_ids:
            continue
        if args.repo and t.repo.split("/")[-1] != args.repo:
            continue
        if t.instance_id in done:
            continue
        if not (data / "snapshots" / f"{t.instance_id}.tgz").exists():
            continue
        todo.append(t)
    print(f"[gold_check] {len(todo)} tasks to verify ({len(done)} already done)")

    def one(t):
        t0 = time.perf_counter()
        env = TaskEnv(data_dir=data, results_dir=out.parent / t.instance_id, instance_id=t.instance_id)
        try:
            r = env.verify(t.patch)
        except Exception as e:  # noqa: BLE001
            r = {"instance_id": t.instance_id, "repo": t.repo, "resolved": False, "error": f"{type(e).__name__}: {e}"}
        tail = r.pop("test_output_tail", "") or ""
        err_line = next((ln for ln in tail.splitlines() if ln.startswith("E ") or "Error" in ln), "")
        rec = {**r, "seconds": round(time.perf_counter() - t0, 1), "first_error": err_line[:200]}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[gold_check] {t.instance_id}: resolved={rec['resolved']} ({rec['seconds']}s) {err_line[:100]}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, todo))

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    by_repo: dict[str, list[int]] = {}
    for r in rows:
        by_repo.setdefault(r["repo"], [0, 0])
        by_repo[r["repo"]][0] += int(bool(r["resolved"]))
        by_repo[r["repo"]][1] += 1
    print("[gold_check] summary:", {k: f"{v[0]}/{v[1]}" for k, v in by_repo.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
