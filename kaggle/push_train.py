"""Upload SFT data (+ training script) as dataset tuxmx1/kagle-sft and push the training kernel.

  python kaggle/push_train.py --name e4b-r16 --episodes data/sft/episodes.jsonl --r 16 --epochs 2
  python kaggle/push_train.py --name e4b-r16 --status --wait     # fetch adapter into runs/kaggle/train-<name>/output
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "tuxmx1"
DATASET = f"{USER}/kagle-sft"


def sh(cmd: list[str]) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, text=True, capture_output=True)
    print("\n".join(l for l in (r.stdout + r.stderr).splitlines() if "NotOpenSSL" not in l and "warnings.warn" not in l).strip())
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", default=str(ROOT / "data/sft/episodes.jsonl"))
    ap.add_argument("--tools", default=str(ROOT / "distill/tools_schema.json"))
    ap.add_argument("--base", default="unsloth/gemma-4-E4B-it")
    ap.add_argument("--base-model-name", default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-seq-len", type=int, default=24576)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--holdout-repo", default="")
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()

    stage = ROOT / "runs" / "kaggle" / f"train-{args.name}"
    stage.mkdir(parents=True, exist_ok=True)
    ref = f"{USER}/kagle-train-{args.name}"
    if args.status:
        while True:
            r = sh(["kaggle", "kernels", "status", ref])
            txt = (r.stdout + r.stderr).lower()
            if not args.wait or ("running" not in txt and "queued" not in txt):
                break
            time.sleep(180)
        out = stage / "output"
        out.mkdir(exist_ok=True)
        sh(["kaggle", "kernels", "output", ref, "-p", str(out)])
        print(sorted(p.name for p in out.rglob("*") if p.is_file())[:30])
        return 0

    if not args.skip_dataset:
        ds = stage / "dataset"
        shutil.rmtree(ds, ignore_errors=True)
        ds.mkdir(parents=True)
        shutil.copy(args.episodes, ds / "episodes.jsonl")
        shutil.copy(args.tools, ds / "tools_schema.json")
        shutil.copy(ROOT / "train" / "train_qlora.py", ds / "train_qlora.py")
        (ds / "dataset-metadata.json").write_text(json.dumps({"title": "kagle-sft", "id": DATASET, "licenses": [{"name": "Apache 2.0"}]}))
        # `datasets status` can 403 on an existing dataset, so try a new version first and only create if that fails
        r = sh(["kaggle", "datasets", "version", "-p", str(ds), "-m", f"sft {time.strftime('%Y-%m-%d %H:%M')}"])
        if "uccessful" not in (r.stdout + r.stderr):
            sh(["kaggle", "datasets", "create", "-p", str(ds)])
        for _ in range(20):
            if "ready" in (sh(["kaggle", "datasets", "status", DATASET]).stdout or "").lower():
                break
            time.sleep(15)

    kd = stage / "kernel"
    shutil.rmtree(kd, ignore_errors=True)
    kd.mkdir(parents=True)
    src = (ROOT / "kaggle" / "train_kernel.py").read_text()
    subs = {
        "BASE": json.dumps(args.base), "BASE_MODEL_NAME": json.dumps(args.base_model_name),
        "R": str(args.r), "ALPHA": str(args.alpha or args.r), "LR": repr(args.lr), "EPOCHS": repr(args.epochs),
        "MAX_SEQ_LEN": str(args.max_seq_len), "GRAD_ACCUM": str(args.grad_accum), "HOLDOUT_REPO": json.dumps(args.holdout_repo),
    }
    for k, v in subs.items():
        src = re.sub(rf"^{k} = .*$", f"{k} = {v}", src, flags=re.M)
    (kd / "train_kernel.py").write_text(src)
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": ref, "title": f"kagle train {args.name}", "code_file": "train_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [], "model_sources": [],
        "machine_shape": "NvidiaL4",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    }, indent=2))
    cmd = ["kaggle", "kernels", "push", "-p", str(kd), "--accelerator", "NvidiaL4"]
    if args.no_run:
        cmd.append("--no-run")
    sh(cmd)
    print(f"[push_train] {ref} pushed; poll: python kaggle/push_train.py --name {args.name} --status --wait")
    return 0


if __name__ == "__main__":
    sys.exit(main())
