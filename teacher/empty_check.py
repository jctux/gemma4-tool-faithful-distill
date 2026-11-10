"""Control: an EMPTY patch must FAIL verification. Tasks that pass without a fix are unsound.
  python -m teacher.empty_check --data-dir /mnt/nvme/kaggle/gemma4 --task-ids-file eval/heldout_tasks.json --out runs/gold/empty_control.jsonl --workers 2
"""
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teacher.harness_env import TaskEnv

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--task-ids-file", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()
ids = json.load(open(a.task_ids_file)); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)["instance_id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()

def one(t):
    t0 = time.time()
    try:
        r = TaskEnv(data_dir=Path(a.data_dir), results_dir=out.parent / "empty" / t, instance_id=t).verify("")
        rec = {"instance_id": t, "empty_resolved": bool(r["resolved"]), "exit": r.get("test_exit_code"), "error": r.get("error"), "seconds": round(time.time() - t0)}
    except Exception as e:
        rec = {"instance_id": t, "empty_resolved": None, "error": f"{type(e).__name__}: {e}"}
    open(out, "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)

with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(one, [t for t in ids if t not in done]))
print("EMPTY_CONTROL_DONE")
