"""Run one teacher (Claude Code) trajectory for a task against the remote harness sandbox.

Flow (Mac side):
  1. rsync repo code to the sandbox host
  2. start teacher/mcp_server.py there (background), wait for READY, fetch prompt.txt
  3. ssh port-forward, launch `claude -p` restricted to the harness MCP tools
  4. SIGTERM the server → patch.diff + status.json; fetch artifacts
  5. run teacher/verify.py remotely → verify.json; append summary to runs/teacher/index.jsonl

  python teacher/run_task.py --task-id httpx_3672 [--sample 0] [--model claude-opus-5-5]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILTIN_TOOLS = (
    "Bash,Read,Edit,Write,MultiEdit,Glob,Grep,LS,WebFetch,WebSearch,Task,Agent,"
    "NotebookEdit,NotebookRead,TodoWrite,ToolSearch,KillShell,BashOutput,EnterPlanMode,ExitPlanMode"
)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, text=True, capture_output=True, **kw)


def ssh(host: str, remote_cmd: str, **kw) -> subprocess.CompletedProcess:
    return sh(["ssh", "-o", "BatchMode=yes", host, remote_cmd], **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--sample", type=int, default=0, help="sample index (results go to <task>/s<sample>)")
    ap.add_argument("--host", default=os.environ.get("GPU_HOST", "user@gpu-host"))
    ap.add_argument("--remote-repo", default="/home/jc/kagle")
    ap.add_argument("--remote-data", default="/mnt/nvme/kaggle/gemma4")
    ap.add_argument("--remote-docker-host", default="unix:///run/user/1000/podman/podman.sock")
    ap.add_argument("--results", default=str(ROOT / "runs" / "teacher"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-turns", type=int, default=80, help="Claude Code turn cap")
    ap.add_argument("--max-tool-calls", type=int, default=100)
    ap.add_argument("--max-time-minutes", type=float, default=60.0)
    ap.add_argument("--system-prompt", default=str(ROOT / "submission" / "prompts" / "system.md"))
    ap.add_argument("--setup-timeout", type=int, default=900)
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    run_tag = f"{args.task_id}/s{args.sample}"
    local_dir = Path(args.results) / run_tag
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_results = f"{args.remote_repo}/runs/teacher/{args.task_id}/s{args.sample}"
    remote_out = f"{remote_results}/{args.task_id}"  # mcp_server nests <results>/<task>

    if not args.no_sync:
        r = sh([str(ROOT / "scripts" / "sync.sh")])
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            return 2

    # 1. start server
    server_cmd = (
        f"cd {args.remote_repo} && mkdir -p {remote_results} && "
        f"DOCKER_HOST={args.remote_docker_host} nohup .venv/bin/python -m teacher.mcp_server "
        f"--task-id {shlex.quote(args.task_id)} --data-dir {shlex.quote(args.remote_data)} "
        f"--results-dir {shlex.quote(remote_results)} --port {args.port} "
        f"--max-tool-calls {args.max_tool_calls} --max-time-minutes {args.max_time_minutes} "
        f"> {remote_results}/server.log 2>&1 & echo $!"
    )
    r = ssh(args.host, server_cmd)
    if r.returncode != 0:
        print("server start failed:", r.stderr, file=sys.stderr)
        return 2
    print(f"[run_task] server pid {r.stdout.strip()} on {args.host}:{args.port}")

    # 2. wait READY
    t0 = time.time()
    prompt: str | None = None
    while time.time() - t0 < args.setup_timeout:
        r = ssh(args.host, f"cat {remote_out}/READY 2>/dev/null && cat {remote_out}/prompt.txt")
        if r.returncode == 0 and r.stdout.strip():
            prompt = r.stdout.split("\n", 1)[1] if "\n" in r.stdout else ""
            break
        r2 = ssh(args.host, f"cat {remote_out}/status.json 2>/dev/null")
        if r2.stdout.strip():
            print("setup failed:", r2.stdout, file=sys.stderr)
            print(ssh(args.host, f"tail -40 {remote_results}/server.log").stdout, file=sys.stderr)
            return 3
        time.sleep(5)
    if prompt is None:
        print("timed out waiting for sandbox", file=sys.stderr)
        ssh(args.host, f"kill $(cat {remote_out}/server.pid) 2>/dev/null")
        return 3
    (local_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"[run_task] sandbox ready in {time.time() - t0:.0f}s; prompt {len(prompt)} chars")

    # 3. port-forward + claude
    tunnel = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", "-N", "-L", f"{args.port}:127.0.0.1:{args.port}", args.host]
    )
    time.sleep(2)
    mcp_cfg = local_dir / "mcp.json"
    mcp_cfg.write_text(
        json.dumps({"mcpServers": {"harness": {"type": "http", "url": f"http://127.0.0.1:{args.port}/mcp"}}}),
        encoding="utf-8",
    )
    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    (local_dir / "system_prompt.md").write_text(system_prompt, encoding="utf-8")  # exact prompt used, for to_sft
    claude_cmd = [
        "claude", "-p", prompt,
        "--system-prompt", system_prompt,
        "--mcp-config", str(mcp_cfg), "--strict-mcp-config",
        "--tools", "",  # disable every built-in tool; only the harness MCP tools remain
        "--allowedTools", "mcp__harness__*",
        "--disallowedTools", BUILTIN_TOOLS,
        "--permission-mode", "bypassPermissions",
        "--max-turns", str(args.max_turns),
        "--output-format", "stream-json", "--verbose",
    ]
    if args.model:
        claude_cmd += ["--model", args.model]
    (local_dir / "claude_cmd.txt").write_text(" ".join(shlex.quote(c) for c in claude_cmd), encoding="utf-8")
    t1 = time.time()
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}  # allow nested headless run
    with open(local_dir / "claude_stream.jsonl", "w", encoding="utf-8") as out:
        proc = subprocess.run(claude_cmd, stdout=out, stderr=subprocess.PIPE, text=True, env=env, cwd=str(local_dir))
    (local_dir / "claude_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    print(f"[run_task] claude exited {proc.returncode} after {time.time() - t1:.0f}s")

    # 4. finalize server, fetch artifacts
    ssh(args.host, f"kill -TERM $(cat {remote_out}/server.pid) 2>/dev/null; sleep 3")
    for _ in range(20):
        if ssh(args.host, f"test -f {remote_out}/status.json").returncode == 0:
            break
        time.sleep(2)
    tunnel.terminate()
    sh(["scp", "-q", "-r", f"{args.host}:{remote_out}/.", str(local_dir)])
    sh(["scp", "-q", f"{args.host}:{remote_results}/server.log", str(local_dir / "server.log")])
    status = json.loads((local_dir / "status.json").read_text()) if (local_dir / "status.json").exists() else {}
    print("[run_task] status:", json.dumps(status))

    # 5. verify
    verify: dict = {}
    if not args.skip_verify:
        r = ssh(
            args.host,
            f"cd {args.remote_repo} && DOCKER_HOST={args.remote_docker_host} .venv/bin/python -m teacher.verify "
            f"--task-id {shlex.quote(args.task_id)} --data-dir {shlex.quote(args.remote_data)} "
            f"--patch {remote_out}/patch.diff --out {remote_out}/verify.json "
            f"> {remote_results}/verify.log 2>&1; cat {remote_out}/verify.json",
        )
        try:
            verify = json.loads(r.stdout)
        except json.JSONDecodeError:
            verify = {"resolved": False, "error": "verify_failed", "raw": r.stdout[-2000:], "stderr": r.stderr[-2000:]}
        (local_dir / "verify.json").write_text(json.dumps(verify, indent=2), encoding="utf-8")
        print("[run_task] verify:", json.dumps({k: v for k, v in verify.items() if k != "test_output_tail"}))

    summary = {
        "task_id": args.task_id,
        "sample": args.sample,
        "model": args.model,
        "claude_exit": proc.returncode,
        "claude_seconds": round(time.time() - t1),
        "patch_chars": status.get("patch_chars"),
        "tool_calls_used": status.get("tool_calls_used"),
        "resolved": verify.get("resolved"),
        "verify_error": verify.get("error"),
        "ts": time.time(),
    }
    with open(Path(args.results) / "index.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print("[run_task] summary:", json.dumps(summary))
    return 0 if verify.get("resolved") else 1


if __name__ == "__main__":
    sys.exit(main())
