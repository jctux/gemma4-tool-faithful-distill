"""Convert teacher runs (runs/teacher/<task>/s<k>/) into SFT records.

Each record = one full episode as OpenAI-style messages:
  system  : the shipped system prompt (submission/prompts/system.md at run time)
  user    : harness task prompt (prompt.txt)
  assistant: {content: <short visible text / thought>, tool_calls: [...]}
  tool    : harness JSON result (string) for each call
  ...
  assistant: final text

Source of truth for the assistant side is Claude Code's stream-json transcript
(claude_stream.jsonl); tool results come from the harness log (trajectory.jsonl) matched by
order, falling back to the tool_result blocks in the transcript.

  python -m distill.to_sft --runs runs/teacher --out data/sft/episodes.jsonl [--only-resolved]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

MCP_PREFIX = "mcp__harness__"


def _load_jsonl(p: Path) -> list[dict[str, Any]]:
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _tool_result_text(block: Any) -> str:
    """Flatten a Claude tool_result content block into the raw JSON string the tool returned."""
    content = block.get("content") if isinstance(block, dict) else block
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def episode_from_run(run_dir: Path, system_prompt: str, max_thought_chars: int = 1200) -> dict[str, Any] | None:
    """Rebuild API-level turns from Claude Code's stream (text/tool_use blocks and tool results are
    streamed as separate events, pipelined, and can interleave across parallel tool calls)."""
    stream_p = run_dir / "claude_stream.jsonl"
    prompt_p = run_dir / "prompt.txt"
    if not stream_p.exists() or not prompt_p.exists():
        return None
    events = _load_jsonl(stream_p)
    harness = [r for r in _load_jsonl(run_dir / "trajectory.jsonl") if "tool" in r] if (run_dir / "trajectory.jsonl").exists() else []
    sp_run = run_dir / "system_prompt.md"
    if sp_run.exists():
        system_prompt = sp_run.read_text(encoding="utf-8")  # prompt actually used for this run

    turns: list[dict[str, Any]] = []          # assistant turns: {content, tool_calls}
    results: dict[str, str] = {}              # tool_use id → raw result text (from stream)
    call_owner: dict[str, int] = {}           # tool_use id → turn index
    n_calls = 0

    def turn_has_result(t: dict[str, Any]) -> bool:
        return any(tc["id"] in results for tc in t.get("tool_calls", []))

    for ev in events:
        if ev.get("type") == "assistant":
            text_parts, tool_calls = [], []
            for block in ev.get("message", {}).get("content", []):
                btype = block.get("type")
                if btype == "text" and block.get("text", "").strip():
                    text_parts.append(block["text"].strip())
                elif btype == "thinking" and block.get("thinking", "").strip():
                    text_parts.append(block["thinking"].strip())
                elif btype == "tool_use":
                    name = block.get("name", "")
                    if name.startswith(MCP_PREFIX):
                        name = name[len(MCP_PREFIX):]
                    cid = block.get("id", f"call_{n_calls}")
                    n_calls += 1
                    tool_calls.append({"id": cid, "type": "function",
                                       "function": {"name": name, "arguments": dict(block.get("input", {}) or {})}})  # mapping: Gemma 4 template requires it
            if not text_parts and not tool_calls:
                continue
            content = "\n".join(text_parts)
            # a new API message starts once the previous turn's tool calls began returning results
            if turns and not (turns[-1].get("tool_calls") and turn_has_result(turns[-1])):
                t = turns[-1]
                t["content"] = (t["content"] + "\n" + content).strip() if content else t["content"]
                t.setdefault("tool_calls", []).extend(tool_calls)
            else:
                turns.append({"content": content, "tool_calls": tool_calls})
            for tc in tool_calls:
                call_owner[tc["id"]] = len(turns) - 1
        elif ev.get("type") == "user":
            content = ev.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        results[block.get("tool_use_id", "")] = _tool_result_text(block)

    # prefer the harness-side record (exact JSON returned by swegemma), matched by tool name + args
    pending = list(harness)

    def harness_result(name: str, arguments: dict[str, Any] | str) -> str | None:
        args = arguments if isinstance(arguments, dict) else None
        for i, rec in enumerate(pending):
            if rec.get("tool") == name and (args is None or {k: v for k, v in rec.get("args", {}).items() if v is not None} == {k: v for k, v in (args or {}).items() if v is not None}):
                pending.pop(i)
                return json.dumps(rec.get("result"), ensure_ascii=False)
        return None

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_p.read_text(encoding="utf-8")},
    ]
    for t in turns:
        content = t["content"]
        if len(content) > max_thought_chars:
            content = content[: max_thought_chars - 1] + "…"
        m: dict[str, Any] = {"role": "assistant", "content": content}
        if t.get("tool_calls"):
            # Gemma 4 template renders `reasoning_content` as <|channel>thought…<channel|> before the
            # tool call (when preserve_thinking=True); plain content is dropped on tool-call turns.
            m = {"role": "assistant", "content": "", "reasoning_content": content, "tool_calls": t["tool_calls"]}
        messages.append(m)
        for tc in t.get("tool_calls", []):
            res = harness_result(tc["function"]["name"], tc["function"]["arguments"])
            if res is None:
                res = results.get(tc["id"], json.dumps({"status": "error", "error_type": "MissingResult", "error_message": "no result recorded"}))
            messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"], "content": res})
    if messages[-1]["role"] != "assistant":
        messages.append({"role": "assistant", "content": "Done."})

    status = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
    verify = json.loads((run_dir / "verify.json").read_text()) if (run_dir / "verify.json").exists() else {}
    m = re.match(r"(.+)/s(\d+)$", run_dir.as_posix())
    return {
        "task_id": run_dir.parent.name,
        "sample": int(m.group(2)) if m else 0,
        "resolved": bool(verify.get("resolved")),
        "n_tool_calls": n_calls,
        "n_messages": len(messages),
        "patch_chars": status.get("patch_chars"),
        "messages": messages,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/teacher")
    ap.add_argument("--out", default="data/sft/episodes.jsonl")
    ap.add_argument("--system-prompt", default="submission/prompts/system.md")
    ap.add_argument("--only-resolved", action="store_true")
    ap.add_argument("--max-thought-chars", type=int, default=1200)
    args = ap.parse_args()

    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = kept = 0
    with open(out, "w", encoding="utf-8") as f:
        for run_dir in sorted(Path(args.runs).glob("*/s*")):
            ep = episode_from_run(run_dir, system_prompt, args.max_thought_chars)
            if ep is None:
                continue
            n += 1
            if args.only_resolved and not ep["resolved"]:
                continue
            kept += 1
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    print(f"episodes scanned={n} written={kept} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
