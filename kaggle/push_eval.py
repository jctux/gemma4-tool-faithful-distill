"""Push (1) the submission dir as a Kaggle dataset version and (2) the eval kernel, then optionally wait.

  python kaggle/push_eval.py --name e4b-base --repo Textualize/rich --max-tasks 20
  python kaggle/push_eval.py --name e4b-lora1 --task-ids rich_3061 rich_4077 --no-run   # just save
  python kaggle/push_eval.py --status e4b-base                                         # poll + fetch output
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
DATASET = f"{USER}/kagle-submission"
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
MODEL_SOURCES = {
    "gemma-4-e2b-it-qat-w4a16-ct": "google/gemma-4/Other/gemma-4-e2b-it-qat-w4a16-ct/2",
    "gemma-4-e4b-it-qat-w4a16-ct": "google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct/2",
    "gemma-4-12b-it-qat-w4a16-ct": "google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2",
    "gemma-4-31b-it-qat-w4a16-ct": "google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2",
}


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, text=True, capture_output=True, **kw)
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "NotOpenSSL" not in l and "warnings.warn" not in l)
    print(out.strip())
    return r


def push_dataset(sub_dir: Path, stage: Path) -> None:
    ds = stage / "dataset"
    shutil.rmtree(ds, ignore_errors=True)
    shutil.copytree(sub_dir, ds / "submission", ignore=shutil.ignore_patterns("README.md", "*.pyc", "__pycache__"))
    (ds / "dataset-metadata.json").write_text(json.dumps({
        "title": "kagle-submission", "id": DATASET, "licenses": [{"name": "Apache 2.0"}]}, indent=2))
    exists = sh(["kaggle", "datasets", "status", DATASET]).returncode == 0
    if exists:
        sh(["kaggle", "datasets", "version", "-p", str(ds), "-m", f"update {time.strftime('%Y-%m-%d %H:%M')}", "--dir-mode", "zip"])
    else:
        sh(["kaggle", "datasets", "create", "-p", str(ds), "--dir-mode", "zip"])


def push_kernel(name: str, stage: Path, task_ids: list[str], repo: str, max_tasks: int, concurrency: int,
                model: str, no_run: bool, accelerator: str = "NvidiaL4") -> str:
    kd = stage / "kernel"
    shutil.rmtree(kd, ignore_errors=True)
    kd.mkdir(parents=True)
    src = (ROOT / "kaggle" / "eval_kernel.py").read_text()
    src = re.sub(r"^TASK_IDS: list\[str\] = .*$", f"TASK_IDS: list[str] = {json.dumps(task_ids)}", src, flags=re.M)
    src = re.sub(r'^REPO = .*$', f"REPO = {json.dumps(repo)}", src, flags=re.M)
    src = re.sub(r"^MAX_TASKS = .*$", f"MAX_TASKS = {max_tasks}", src, flags=re.M)
    src = re.sub(r"^CONCURRENCY = .*$", f"CONCURRENCY = {concurrency}", src, flags=re.M)
    (kd / "eval_kernel.py").write_text(src)
    slug = f"kagle-eval-{name}"
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}", "title": f"kagle eval {name}", "code_file": "eval_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", DATASET],
        "competition_sources": ["gemma-4-developer-agent"], "kernel_sources": [],
        "model_sources": [MODEL_SOURCES[model]], "machine_shape": accelerator,
        # same Python 3.12 image as the official starter notebook (wheelhouse is cp312)
        "docker_image": DOCKER_IMAGE,
    }, indent=2))
    cmd = ["kaggle", "kernels", "push", "-p", str(kd), "--accelerator", accelerator]  # metadata machine_shape alone ran on CPU
    if no_run:
        cmd.append("--no-run")
    sh(cmd)
    return f"{USER}/{slug}"


def status(name: str, out_dir: Path, wait: bool) -> None:
    ref = f"{USER}/kagle-eval-{name}"
    while True:
        r = sh(["kaggle", "kernels", "status", ref])
        if not wait or "running" not in (r.stdout + r.stderr).lower() and "queued" not in (r.stdout + r.stderr).lower():
            break
        time.sleep(120)
    out_dir.mkdir(parents=True, exist_ok=True)
    sh(["kaggle", "kernels", "output", ref, "-p", str(out_dir)])
    s = out_dir / "eval_summary.json"
    if s.exists():
        print(s.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--submission", default=str(ROOT / "submission"))
    ap.add_argument("--task-ids", nargs="*", default=[])
    ap.add_argument("--repo", default="")
    ap.add_argument("--max-tasks", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default=None, help="base model alias; default: read from submission/agent.yaml")
    ap.add_argument("--accelerator", default="NvidiaL4", help="e.g. NvidiaL4, NvidiaL4X4, NvidiaTeslaT4")
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--status", action="store_true", help="only poll status + fetch outputs")
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()

    stage = ROOT / "runs" / "kaggle" / args.name
    stage.mkdir(parents=True, exist_ok=True)
    if args.status:
        status(args.name, stage / "output", args.wait)
        return 0
    model = args.model
    if model is None:
        m = re.search(r"^model:\s*(\S+)", (Path(args.submission) / "agent.yaml").read_text(), re.M)
        model = m.group(1)
    if not args.skip_dataset:
        push_dataset(Path(args.submission), stage)
    ref = push_kernel(args.name, stage, args.task_ids, args.repo, args.max_tasks, args.concurrency, model, args.no_run, args.accelerator)
    print(f"[push_eval] kernel {ref} pushed; poll with: python kaggle/push_eval.py --status {args.name} --wait")
    if args.wait:
        status(args.name, stage / "output", True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
