"""Local evaluation that mirrors the Kaggle starter notebook (Evaluator + compaction + cache config).

  OPENAI_BASE_URL=http://127.0.0.1:8000/v1 DOCKER_HOST=unix:///run/user/1000/podman/podman.sock \
  python eval/run_eval.py --submission eval_subs/base --name base --tasks-file eval/heldout_tasks.json --concurrency 2
Uses teacher.harness_env's pin-aware dependency fix for BOTH the agent sandbox and verification.
"""
import argparse, asyncio, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import teacher.harness_env as he  # noqa: F401  (patches verification deps)
import swegemma.harness.agent_runner as ar
ar.install_test_dependencies = he.install_test_dependencies_pinned  # agent sandbox bootstrap too

import yaml
from adk_submission import discover_adapters
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.cli import _build_cli_model_registry
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator

ap = argparse.ArgumentParser()
ap.add_argument("--submission", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--data-dir", default="/mnt/nvme/kaggle/gemma4")
ap.add_argument("--tasks-file", default=str(ROOT / "eval/heldout_tasks.json"))
ap.add_argument("--task-ids", nargs="*", default=None)
ap.add_argument("--concurrency", type=int, default=2)
ap.add_argument("--results-root", default=str(ROOT / "runs/eval"))
ap.add_argument("--no-compaction", action="store_true")
a = ap.parse_args()

sub = Path(a.submission).resolve(); data = Path(a.data_dir)
ids = a.task_ids or json.load(open(a.tasks_file))
ev = (yaml.safe_load((sub / "eval_config.yaml").read_text()) or {}).get("evaluation", {})
limits, gen = build_submission_limits()
adapters = discover_adapters(str(sub), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
models = _build_cli_model_registry(submission_dir=sub)
cfg = EvalConfig(
    tasks_path=data / "tasks.jsonl", snapshots_dir=data / "snapshots", results_dir=Path(a.results_root) / a.name,
    submission_dir=sub, models=models, sandbox="docker",
    timeout_seconds=int(ev.get("timeout_seconds", 300)), verification_timeout_seconds=int(ev.get("verification_timeout_seconds", 300)),
    max_time_minutes=float(ev.get("max_time_minutes", 60)), max_tool_calls=int(ev.get("max_tool_calls", 100)),
    max_turns=int(ev["max_turns"]) if ev.get("max_turns") else None, task_ids=ids, limits=limits, generation_constraints=gen,
    adapter_manifest=adapters,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=None if a.no_compaction else EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
    graph_dir=str(data / "graphs"), embeddings_dir=str(data / "embeddings"), wheels_dir=data / "wheels",
    concurrency=a.concurrency, display_mode="quiet", verbose=False,
)
t0 = time.time()
res = asyncio.run(Evaluator(cfg).run())
dt = time.time() - t0
out = {"name": a.name, "resolved": res.resolved, "total": res.total, "rate": res.resolution_rate, "seconds": round(dt), "s_per_task": round(dt / max(1, res.total)), "tasks": ids}
Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
(Path(cfg.results_dir) / "eval_summary.json").write_text(json.dumps(out, indent=1))
print("EVAL_DONE", json.dumps({k: v for k, v in out.items() if k != "tasks"}))
