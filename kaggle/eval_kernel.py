"""Kaggle script kernel: evaluate a submission directory with the official harness on L4x4.

Derived from ryanholbrook/getting-started-gemma-4-developer-agent (official starter).
Inputs (attached in kernel-metadata.json):
  - competition data           /kaggle/input/competitions/gemma-4-developer-agent
  - wheelhouse dataset         /kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse
  - model                      /kaggle/input/models/google/gemma-4/other/<alias>/2
  - our submission dataset     /kaggle/input/datasets/tuxmx1/kagle-submission  (agent.yaml, prompts, adapters…)
Config via the SUBMISSION / TASK_IDS / MAX_TASKS constants below (edited by kaggle/push_eval.py).
Outputs: /kaggle/working/results/summary.json, task_results.jsonl, patches/, predictions.csv
"""

import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- config (patched by push_eval.py)
SUBMISSION_DIR_NAME = "submission"      # subdir inside the attached submission dataset
TASK_IDS: list[str] = []                # empty → all tasks (filtered by REPO / MAX_TASKS)
REPO = ""                               # e.g. "Textualize/rich"
MAX_TASKS = 20
CONCURRENCY = 4
# ------------------------------------------------------------------------------------------------

os.environ.update({
    "TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_FLAX": "1", "TRANSFORMERS_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_ENGINE_READY_TIMEOUT_S": "1200", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
    "NCCL_DEBUG": "WARN", "LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_TELEMETRY": "False",
    "LITELLM_LOG": "WARNING", "OPENAI_API_KEY": "EMPTY", "LOCAL_API_KEY": "EMPTY",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OTEL_SDK_DISABLED": "true",
    "OTEL_PYTHON_DISABLED": "true",
})
for name in ("LiteLLM", "litellm", "httpx", "httpcore", "openai", "urllib3", "opentelemetry"):
    logging.getLogger(name).setLevel(logging.WARNING)

WHEELHOUSE_DIR = Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse")
DATA_DIR = Path("/kaggle/input/competitions/gemma-4-developer-agent")
WORKING_DIR = Path("/kaggle/working")
RESULTS_DIR = WORKING_DIR / "results"
def _find_submission_dataset() -> Path | None:
    """Locate our attached submission dataset regardless of Kaggle's mount layout."""
    for cand in sorted(Path("/kaggle/input").rglob("agent.yaml")):
        s = cand.as_posix()
        if "/competitions/" in s or "sample_submission" in s:
            continue
        return cand.parent
    return None


print("[inputs]", {p.name: sorted(x.name for x in p.iterdir())[:8] for p in Path("/kaggle/input").iterdir() if p.is_dir()})
SUB_DS = _find_submission_dataset()

# ---------------------------------------------------------------- 1. install harness from wheelhouse
for pth in glob.glob("/usr/local/lib/python*/*-packages/*cutlass*.pth"):
    try:
        os.unlink(pth)
    except OSError:
        pass
tmp_whl = Path("/tmp/wheelhouse")
tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob("*.whl"):
    if "cutlass" in w.name.lower():
        continue
    target = tmp_whl / (w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name)
    if not target.exists():
        try:
            os.symlink(w, target)
        except OSError:
            shutil.copyfile(w, target)
wheels = sorted(str(w) for w in tmp_whl.glob("*.whl"))
t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *wheels], check=True)  # same as official starter
print(f"[setup] harness wheels installed in {time.time() - t0:.0f}s")

import litellm  # noqa: E402
import torch  # noqa: E402
from adk_submission import VllmConfig, VllmServer, discover_adapters  # noqa: E402
from adk_submission.discovery import discover_declared_models  # noqa: E402
from google.adk.agents.context_cache_config import ContextCacheConfig  # noqa: E402
from google.adk.apps._configs import EventsCompactionConfig  # noqa: E402
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402
from swegemma.models import load_tasks  # noqa: E402

litellm.drop_params = True

# ---------------------------------------------------------------- 2. submission + tasks
assert SUB_DS is not None, "attach the tuxmx1/kagle-submission dataset (no agent.yaml found under /kaggle/input)"
AGENT_SRC = SUB_DS
print("[submission]", AGENT_SRC, sorted(p.name for p in AGENT_SRC.iterdir()))
AGENT_DIR = WORKING_DIR / "agent"
if AGENT_DIR.exists():
    shutil.rmtree(AGENT_DIR)
shutil.copytree(AGENT_SRC, AGENT_DIR)
for junk in ("README.md",):
    (AGENT_DIR / junk).unlink(missing_ok=True)

tasks = load_tasks(DATA_DIR / "tasks.jsonl")
if TASK_IDS:
    tasks = [t for t in tasks if t.instance_id in set(TASK_IDS)]
if REPO:
    tasks = [t for t in tasks if t.repo == REPO]
tasks = tasks[:MAX_TASKS]
print(f"[tasks] {len(tasks)} tasks: {[t.instance_id for t in tasks][:10]}…")

# ---------------------------------------------------------------- 3. vLLM
declared = discover_declared_models(AGENT_DIR)
assert len(declared) == 1, declared
MODEL = next(iter(declared))
MODEL_PATH = Path(f"/kaggle/input/models/google/gemma-4/other/{MODEL}/2")
assert MODEL_PATH.exists(), MODEL_PATH
adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
adapter_map = getattr(adapters, "adapters", None) or {}
ranks = [r for i in adapter_map.values() if (r := getattr(i, "rank", None))]
gpu_count = torch.cuda.device_count()
tp = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
extra = {"gemma-4-12b-it-qat-w4a16-ct": ("--hf-overrides", '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')}
cfg = VllmConfig(
    model=str(MODEL_PATH), port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
    max_model_len=32768, dtype="bfloat16", gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
    enable_lora=bool(adapter_map), max_loras=max(1, len(adapter_map)), max_lora_rank=max(ranks) if ranks else 128,
    scheduler_reserve_full_isl=False, tensor_parallel_size=tp, startup_timeout=60 * 20, extra_args=extra.get(MODEL, ()),
)
server = VllmServer(cfg, adapter_manifest=adapters)
t0 = time.time()
server.start()
print(f"[vllm] {MODEL} up on {server.base_url} tp={tp} adapters={list(adapter_map)} in {time.time() - t0:.0f}s")
models = server.create_model_registry(aliases=[MODEL], model_prefix="openai/", api_key="EMPTY")

# ---------------------------------------------------------------- 4. eval
import yaml  # noqa: E402

ev = {}
for name in ("eval_config.yaml", "eval_config.yml"):
    p = AGENT_DIR / name
    if p.is_file():
        ev = (yaml.safe_load(p.read_text()) or {}).get("evaluation", {})
limits, gen_constraints = build_submission_limits()
eval_config = EvalConfig(
    tasks_path=DATA_DIR / "tasks.jsonl", snapshots_dir=DATA_DIR / "snapshots", results_dir=RESULTS_DIR,
    submission_dir=AGENT_DIR, models=models, sandbox="subprocess",
    timeout_seconds=int(ev.get("timeout_seconds", 300)),
    verification_timeout_seconds=int(ev.get("verification_timeout_seconds", 300)),
    max_time_minutes=float(ev.get("max_time_minutes", 60.0)), max_tool_calls=int(ev.get("max_tool_calls", 100)),
    max_turns=int(ev["max_turns"]) if ev.get("max_turns") else None,
    task_ids=[t.instance_id for t in tasks], limits=limits, generation_constraints=gen_constraints,
    adapter_manifest=adapters,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
    graph_dir=str(DATA_DIR / "graphs"), embeddings_dir=str(DATA_DIR / "embeddings"), wheels_dir=DATA_DIR / "wheels",
    concurrency=CONCURRENCY, display_mode="quiet", verbose=False,
)
import asyncio  # noqa: E402

t0 = time.time()
result = asyncio.run(Evaluator(eval_config).run())
elapsed = time.time() - t0
print(f"[eval] resolved {result.resolved}/{result.total} = {result.resolution_rate:.3f} in {elapsed / 60:.1f} min "
      f"({elapsed / max(1, result.total):.0f} s/task)")
summary = {"model": MODEL, "adapters": list(adapter_map), "resolved": result.resolved, "total": result.total,
           "resolution_rate": result.resolution_rate, "elapsed_s": elapsed, "tasks": [t.instance_id for t in tasks],
           "eval_config": ev}
(WORKING_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2))
try:
    server.stop()
except Exception:
    pass
