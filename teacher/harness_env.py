"""Thin wrapper around the official swegemma harness for teacher (Claude Code) runs.

Reuses the exact sandbox bootstrap, the 9 budget-gated tools and the initial
prompt builder from `swegemma`, so teacher trajectories match what the student
(Gemma 4) will see at evaluation time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from adk_eval_core.tracing import SessionTrace
from adk_submission.registry import ModelRegistry
from swegemma.config import EvalConfig
from swegemma.context import SwegemmaContext
from swegemma.harness.agent_runner import (
    build_agent_prompt,
    build_workspace_tree_command,
)
from swegemma.harness.container_setup import (
    extract_snapshot,
    install_editable_package,
    install_test_dependencies,
    setup_baseline_commit,
    setup_container_wheels,
    setup_git_exclude,
    setup_workspace_test_config,
)
import swegemma.harness.verification as _verification
from swegemma.harness.verification import verify_task
from swegemma.models.task import Task, load_tasks
from swegemma.sandbox.docker import ContainerConfig, ContainerManager

logger = logging.getLogger(__name__)

# --- pin-aware dependency resolution -------------------------------------------------------
# The harness injects the *newest* wheel of every package (e.g. starlette 1.6.0) regardless of the
# repo's pins, which breaks older fastapi snapshots (Router(on_startup=...) TypeError, missing
# test deps). After the stock bootstrap we let pip resolve the repo's own pins offline from /wheels.
# Local verification only; disable with KAGLE_PIN_RESOLVE=0.
_PIN_RESOLVE_CMD = r"""
cd /workspace || exit 0
python3 - <<'PYEOF' >/tmp/pin_resolve.log 2>&1
import re, subprocess, sys, tomllib, pathlib
specs = []
pp = pathlib.Path('pyproject.toml')
if pp.exists():
    try:
        data = tomllib.loads(pp.read_text())
        specs += list(data.get('project', {}).get('dependencies', []) or [])
    except Exception as e:
        print('pyproject parse failed:', e)
for name in ('requirements.txt', 'requirements-tests.txt', 'requirements_test.txt', 'requirements-test.txt',
             'test-requirements.txt', 'requirements/test.txt', 'requirements/tests.txt', 'requirements-dev.txt'):
    p = pathlib.Path(name)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.split('#', 1)[0].strip()
            if not line or line.startswith(('-', '.')):
                continue
            specs.append(line)
import glob, shutil, importlib.metadata as md
SP = '/usr/local/lib/python3.13/site-packages'
PIP = [sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--find-links=/wheels',
       '--no-build-isolation', '--disable-pip-version-check']

def satisfied(spec):
    r = subprocess.run([sys.executable, '-c',
        'import sys; from packaging.requirements import Requirement; from importlib.metadata import version, PackageNotFoundError\n'
        'req=Requirement(sys.argv[1])\n'
        'try: v=version(req.name)\nexcept PackageNotFoundError: sys.exit(1)\n'
        'sys.exit(0 if (not req.specifier or req.specifier.contains(v, prereleases=True)) else 2)', spec],
        capture_output=True, text=True)
    return r.returncode == 0

def purge(name):
    # harness-injected packages have no RECORD → pip cannot uninstall them; remove by hand
    n = name.lower().replace('-', '_')
    for p in glob.glob(f'{SP}/{n}') + glob.glob(f'{SP}/{n}-*.dist-info') + glob.glob(f'{SP}/{name}-*.dist-info'):
        shutil.rmtree(p, ignore_errors=True)

seen, ok, bad = set(), [], []
for spec in specs:
    key = re.split(r'[<>=!~\[ ;]', spec, maxsplit=1)[0].lower()
    if not key or key in seen or key in ('mypy', 'ruff', 'black', 'isort', 'coverage', 'pre-commit', 'playwright'):
        continue
    seen.add(key)
    if satisfied(spec):
        ok.append(spec + ' (kept)')
        continue
    purge(key)
    r = subprocess.run(PIP + ['--ignore-installed', '--no-deps', spec], capture_output=True, text=True)
    if r.returncode != 0 and '==' in spec:  # exact test-dep pin not in wheelhouse → take what we have
        r = subprocess.run(PIP + ['--ignore-installed', '--no-deps', key], capture_output=True, text=True)
    (ok if r.returncode == 0 else bad).append(spec)
print('pinned ok:', ok)
print('pinned unavailable:', bad)
PYEOF
tail -n 3 /tmp/pin_resolve.log; true
"""

_orig_install_test_dependencies = install_test_dependencies


def install_test_dependencies_pinned(docker, container_id, repo: str = "", *, fast_path: bool = True, config=None):
    _orig_install_test_dependencies(docker, container_id, repo, fast_path=fast_path, config=config)
    if os.environ.get("KAGLE_PIN_RESOLVE", "1") != "1":
        return
    res = docker.exec(container_id, _PIN_RESOLVE_CMD, timeout=900)
    logger.info("pin_resolve exit=%s tail=%s", res.exit_code, (res.stdout or "")[-300:].replace("\n", " | "))


# verify_task() resolves this name at call time from its own module globals.
_verification.install_test_dependencies = install_test_dependencies_pinned

TOOL_NAMES: tuple[str, ...] = (
    "run_command",
    "read_file",
    "write_file",
    "edit_file",
    "submit_patch",
    "get_status",
    "get_code_neighbors",
    "search_similar_code",
    "get_code_subgraph",
)


def find_task(tasks_path: Path, instance_id: str) -> Task:
    for t in load_tasks(tasks_path):
        if t.instance_id == instance_id:
            return t
    raise KeyError(f"task {instance_id!r} not found in {tasks_path}")


@dataclass
class TaskEnv:
    """One task's sandbox (Container A), bound tools and prompt."""

    data_dir: Path
    results_dir: Path
    instance_id: str
    max_tool_calls: int | None = 100
    max_time_minutes: float | None = 60.0
    max_turns: int | None = 500
    command_timeout_seconds: int = 300
    image: str = "swebench-sandbox:latest"
    submission_dir: Path | None = None

    task: Task = field(init=False)
    config: EvalConfig = field(init=False)
    docker: ContainerManager = field(init=False)
    container_id: str = field(init=False, default="")
    context: SwegemmaContext = field(init=False)
    tools: dict[str, Callable[..., dict[str, Any]]] = field(init=False, default_factory=dict)
    prompt: str = field(init=False, default="")
    workspace_tree: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = self.data_dir / "tasks.jsonl"
        self.task = find_task(tasks_path, self.instance_id)
        self.config = EvalConfig(
            tasks_path=tasks_path,
            snapshots_dir=self.data_dir / "snapshots",
            results_dir=self.results_dir,
            submission_dir=Path(self.submission_dir) if self.submission_dir else self.results_dir,
            models=ModelRegistry(),
            image=self.image,
            sandbox="docker",
            graph_dir=str(self.data_dir / "graphs"),
            embeddings_dir=str(self.data_dir / "embeddings"),
            timeout_seconds=self.command_timeout_seconds,
            max_tool_calls=self.max_tool_calls,
            max_time_minutes=self.max_time_minutes,
            max_turns=self.max_turns,
            display_mode="quiet",
        )
        self.docker = ContainerManager(
            ContainerConfig(image=self.image, timeout_seconds=self.command_timeout_seconds)
        )

    # ------------------------------------------------------------------ setup
    @property
    def snapshot_path(self) -> Path:
        return self.data_dir / "snapshots" / f"{self.instance_id}.tgz"

    def setup(self) -> None:
        """Container A bootstrap, identical to run_agent_sandbox()."""
        t0 = time.perf_counter()
        self.container_id = self.docker.start()
        self.context = SwegemmaContext(
            docker_manager=self.docker,
            container_id=self.container_id,
            problem_statement=self.task.problem_statement,
            hints_text=self.task.hints_text or "",
            task=self.task,
            repo=self.task.repo,
            graph_dir=self.config.graph_dir,
            embeddings_dir=self.config.embeddings_dir,
            budget=self.config.budget,
            harness=self.config.harness,
        )
        setup_container_wheels(self.docker, self.container_id, self.config)
        extract_snapshot(self.docker, self.container_id, self.snapshot_path)
        setup_git_exclude(self.docker, self.container_id)
        install_editable_package(self.docker, self.container_id)
        install_test_dependencies_pinned(self.docker, self.container_id, self.task.repo, config=self.config)
        setup_workspace_test_config(self.docker, self.container_id, repo=self.task.repo)
        setup_baseline_commit(self.docker, self.container_id, "baseline")

        tree_res = self.docker.exec(self.container_id, build_workspace_tree_command())
        self.workspace_tree = tree_res.stdout.strip() if tree_res.exit_code == 0 else ""

        self.tools = self.context.create_tools()
        self.prompt = build_agent_prompt(
            task=self.task,
            config=self.config,
            workspace_tree=self.workspace_tree,
            enable_sandbox_testing=True,
            declared_tools=TOOL_NAMES,
        )
        self.context.start_agent_session()
        logger.info("env ready for %s in %.1fs", self.instance_id, time.perf_counter() - t0)

    # ------------------------------------------------------------------ tools
    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        fn = self.tools[name]
        res = fn(**kwargs)
        if isinstance(res, str):
            try:
                res = json.loads(res)
            except json.JSONDecodeError:
                res = {"status": "ok", "result": res}
        return res

    # ------------------------------------------------------------------ patch
    def extract_patch(self) -> str:
        """Submitted patch, or the harness fallback diff if submit_patch was never called."""
        submitted = getattr(self.context, "submitted_patch", "") or ""
        if submitted.strip():
            return submitted
        cmd = (
            "cd /workspace && git add -N . && "
            "(git diff --binary _swegemma_baseline 2>/dev/null || git diff --binary HEAD)"
        )
        res = self.docker.exec(self.container_id, cmd, timeout=120)
        return res.stdout if res.exit_code == 0 else ""

    def status(self) -> dict[str, Any]:
        return self.call("get_status")

    def teardown(self) -> None:
        if self.container_id:
            try:
                self.docker.stop(self.container_id)
            finally:
                self.container_id = ""

    # ------------------------------------------------------------------ verify
    def verify(self, agent_patch: str) -> dict[str, Any]:
        """Phase 2 verification in a fresh Container B. Returns a JSON-able summary."""
        result = asyncio.run(
            verify_task(
                self.docker,
                self.config,
                self.task,
                self.snapshot_path,
                agent_patch=agent_patch,
                trace=SessionTrace(),
                start_time=time.perf_counter(),
            )
        )
        out_dir = self.results_dir / "test_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{self.instance_id}.log").write_text(result.test_output or "", encoding="utf-8")
        return {
            "instance_id": result.instance_id,
            "repo": result.repo,
            "resolved": bool(result.resolved),
            "test_exit_code": result.test_exit_code,
            "error": result.error,
            "duration_seconds": result.duration_seconds,
            "test_output_tail": (result.test_output or "")[-4000:],
        }
