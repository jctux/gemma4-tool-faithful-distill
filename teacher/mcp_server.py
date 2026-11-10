"""MCP server exposing the 9 swegemma harness tools for one task to a Claude Code teacher.

Runs on the sandbox host (deb-gpu) over streamable HTTP on 127.0.0.1:<port>.
Lifecycle:
  1. start → sandbox bootstrap in a background thread (Container A)
  2. writes <results>/<task>/prompt.txt and READY once the env is up
  3. every tool call is appended to <results>/<task>/trajectory.jsonl
  4. SIGTERM/SIGINT (or the `finalize` call from the runner) → patch.diff + status.json,
     container stopped.

Usage:
  python -m teacher.mcp_server --task-id fastapi_11194 --data-dir /mnt/nvme/kaggle/gemma4 \
      --results-dir runs/teacher --port 8765 [--max-tool-calls 100 --max-time-minutes 60]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teacher.harness_env import TaskEnv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("teacher.mcp")

_env: TaskEnv | None = None
_ready = threading.Event()
_setup_error: str | None = None
_finalized = threading.Event()
_lock = threading.Lock()
_out_dir: Path
_traj_path: Path
_call_seq = 0


def _append(record: dict[str, Any]) -> None:
    with _lock, open(_traj_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _wait_ready(timeout: float = 900.0) -> TaskEnv:
    if not _ready.wait(timeout):
        raise RuntimeError("sandbox not ready after %.0fs" % timeout)
    if _setup_error:
        raise RuntimeError(f"sandbox setup failed: {_setup_error}")
    assert _env is not None
    return _env


def _invoke(name: str, **kwargs: Any) -> dict[str, Any]:
    global _call_seq
    env = _wait_ready()
    t0 = time.perf_counter()
    try:
        result = env.call(name, **kwargs)
    except Exception as e:  # mirror harness ToolErrorPlugin: never abort the session
        result = {"status": "error", "error_type": type(e).__name__, "error": str(e)}
    _call_seq += 1
    _append(
        {
            "seq": _call_seq,
            "ts": time.time(),
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "tool": name,
            "args": kwargs,
            "result": result,
        }
    )
    return result


def _finalize(reason: str) -> None:
    if _finalized.is_set():
        return
    _finalized.set()
    if _env is None or not _ready.is_set() or _setup_error:
        (_out_dir / "status.json").write_text(
            json.dumps({"ok": False, "reason": reason, "setup_error": _setup_error}), encoding="utf-8"
        )
        return
    try:
        patch = _env.extract_patch()
        (_out_dir / "patch.diff").write_text(patch, encoding="utf-8")
        status = {
            "ok": True,
            "reason": reason,
            "instance_id": _env.instance_id,
            "repo": _env.task.repo,
            "patch_submitted": bool(getattr(_env.context, "patch_submitted", False)),
            "patch_chars": len(patch),
            "tool_calls_used": getattr(_env.context, "tool_calls_used", None),
            "agent_elapsed_seconds": getattr(_env.context, "elapsed_seconds", None),
            "calls_logged": _call_seq,
        }
        (_out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        log.info("finalized: %s", status)
    finally:
        try:
            _env.teardown()
        except Exception as e:  # pragma: no cover
            log.warning("teardown failed: %s", e)


def _setup_thread(args: argparse.Namespace) -> None:
    global _env, _setup_error
    try:
        env = TaskEnv(
            data_dir=Path(args.data_dir),
            results_dir=_out_dir,
            instance_id=args.task_id,
            max_tool_calls=args.max_tool_calls,
            max_time_minutes=args.max_time_minutes,
            max_turns=args.max_turns,
            command_timeout_seconds=args.timeout_seconds,
            image=args.image,
        )
        env.setup()
        _env = env
        (_out_dir / "prompt.txt").write_text(env.prompt, encoding="utf-8")
        (_out_dir / "workspace_tree.txt").write_text(env.workspace_tree, encoding="utf-8")
        _append({"seq": 0, "ts": time.time(), "event": "task_prompt", "text": env.prompt})
        (_out_dir / "READY").write_text(str(time.time()), encoding="utf-8")
        log.info("READY %s", args.task_id)
    except Exception as e:
        _setup_error = f"{type(e).__name__}: {e}"
        log.exception("setup failed")
        (_out_dir / "status.json").write_text(
            json.dumps({"ok": False, "reason": "setup_failed", "setup_error": _setup_error}),
            encoding="utf-8",
        )
    finally:
        _ready.set()


def build_server(host: str, port: int) -> FastMCP:
    mcp = FastMCP("harness", host=host, port=port)

    @mcp.tool()
    def run_command(command: str) -> dict[str, Any]:
        """Execute a shell command inside the repository sandbox.

        You can run any shell command (Python scripts, tests, git commands, etc.).
        The command runs in the repository workspace (/workspace). No network access.
        Output is truncated to 5000 characters; single command timeout 300 seconds.
        """
        return _invoke("run_command", command=command)

    @mcp.tool()
    def read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> dict[str, Any]:
        """Read the contents of a file from the repository workspace.

        Line numbers are 1-indexed and inclusive. At most 150 lines / 10000 characters
        are returned per call; use start_line/end_line to page through larger files.
        """
        return _invoke("read_file", filepath=filepath, start_line=start_line, end_line=end_line)

    @mcp.tool()
    def write_file(filepath: str, content: str) -> dict[str, Any]:
        """Create or overwrite a file in the workspace (parent directories are created)."""
        return _invoke("write_file", filepath=filepath, content=content)

    @mcp.tool()
    def edit_file(filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> dict[str, Any]:
        """Replace a contiguous block of text in an existing file.

        old_string must match exactly one location unless allow_multiple is true.
        Matching is exact first, then whitespace-flexible, then regex-tolerant.
        """
        return _invoke(
            "edit_file", filepath=filepath, old_string=old_string, new_string=new_string, allow_multiple=allow_multiple
        )

    @mcp.tool()
    def submit_patch() -> dict[str, Any]:
        """Capture the current working tree modifications as the agent's submission.

        Generates a unified git diff against the baseline commit. Does not count toward
        the tool-call budget. Call it last, after verifying your change.
        """
        return _invoke("submit_patch")

    @mcp.tool()
    def get_status() -> dict[str, Any]:
        """Check active budget details (time, tool calls) and active patch details. Free."""
        return _invoke("get_status")

    @mcp.tool()
    def get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> dict[str, Any]:
        """Discover structural code graph neighbors (callers and callees) for a given code node.

        node is a dotted symbol path (e.g. 'pkg.module.Class.method') or a suffix ('Class.method').
        """
        return _invoke("get_code_neighbors", node=node, edge_type=edge_type, max_neighbors=max_neighbors)

    @mcp.tool()
    def search_similar_code(query: str, k: int = 10) -> dict[str, Any]:
        """Find semantically similar code functions and classes using precomputed graph embeddings.

        query is a symbol name or dotted path (e.g. 'pkg.module.Class.method', 'Class.method', 'parse_header').
        """
        return _invoke("search_similar_code", query=query, k=k)

    @mcp.tool()
    def get_code_subgraph(nodes: list[str]) -> dict[str, Any]:
        """Extract the relationship graph (nodes and connecting edges) for a focal set of code elements."""
        return _invoke("get_code_subgraph", nodes=nodes)

    return mcp


def main() -> int:
    global _out_dir, _traj_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-tool-calls", type=int, default=100)
    ap.add_argument("--max-time-minutes", type=float, default=60.0)
    ap.add_argument("--max-turns", type=int, default=500)
    ap.add_argument("--timeout-seconds", type=int, default=300)
    ap.add_argument("--image", default="swebench-sandbox:latest")
    args = ap.parse_args()

    _out_dir = Path(args.results_dir) / args.task_id
    _out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("READY", "status.json", "patch.diff"):
        (_out_dir / stale).unlink(missing_ok=True)
    _traj_path = _out_dir / "trajectory.jsonl"
    _traj_path.write_text("", encoding="utf-8")
    (_out_dir / "server.pid").write_text(str(os.getpid()), encoding="utf-8")

    def _on_signal(signum: int, _frame: Any) -> None:
        log.info("signal %s → finalize", signum)
        _finalize(f"signal_{signum}")
        os._exit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    threading.Thread(target=_setup_thread, args=(args,), daemon=True, name="sandbox-setup").start()
    server = build_server(args.host, args.port)
    try:
        server.run(transport="streamable-http")
    finally:
        _finalize("server_exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
