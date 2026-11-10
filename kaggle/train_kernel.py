"""Kaggle script kernel: QLoRA-train Gemma 4 on teacher episodes (internet ON, 1× L4 used by Unsloth).

Inputs (attached): dataset tuxmx1/kagle-sft containing
  episodes.jsonl, tools_schema.json, train_qlora.py   (uploaded by kaggle/push_train.py)
Output: /kaggle/working/adapter/{adapter_config.json, adapter_model.safetensors, …} + train_log.txt
Config constants below are patched by push_train.py.
"""

import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- config (patched by push_train.py)
BASE = "unsloth/gemma-4-E4B-it"
BASE_MODEL_NAME = "google/gemma-4-E4B-it-qat-w4a16-ct"   # written into adapter_config for the harness
R = 16
ALPHA = 16
LR = 2e-4
EPOCHS = 2.0
MAX_SEQ_LEN = 24576
GRAD_ACCUM = 8
HOLDOUT_REPO = ""
# ------------------------------------------------------------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
SFT = next((p.parent for p in sorted(Path("/kaggle/input").rglob("episodes.jsonl"))), None)  # mount layout varies
assert SFT is not None, "attach tuxmx1/kagle-sft (episodes.jsonl not found under /kaggle/input)"
OUT = Path("/kaggle/working")

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth", "trl", "datasets", "peft"], check=True)
print(f"[setup] unsloth installed in {time.time() - t0:.0f}s")
subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], check=False)

sys.argv = [
    "train_qlora.py",
    "--episodes", str(SFT / "episodes.jsonl"),
    "--tools", str(SFT / "tools_schema.json"),
    "--base", BASE, "--base-model-name", BASE_MODEL_NAME,
    "--out", str(OUT / "train"),
    "--r", str(R), "--alpha", str(ALPHA), "--lr", str(LR), "--epochs", str(EPOCHS),
    "--max-seq-len", str(MAX_SEQ_LEN), "--grad-accum", str(GRAD_ACCUM),
] + (["--holdout-repo", HOLDOUT_REPO] if HOLDOUT_REPO else [])
runpy.run_path(str(SFT / "train_qlora.py"), run_name="__main__")

# collect the adapter at a stable path and drop the heavy trainer checkpoints
import shutil  # noqa: E402

src = OUT / "train" / "adapter"
dst = OUT / "adapter"
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst)
shutil.rmtree(OUT / "train" / "trainer", ignore_errors=True)
print("[done] adapter files:", sorted(p.name for p in dst.iterdir()))
