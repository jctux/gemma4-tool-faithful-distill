"""Split long teacher episodes into overlapping windows instead of dropping them.

Each window = [system, task prompt] + a run of whole assistant/tool turns that fits the token cap.
Consecutive windows overlap by OVERLAP turns so each model turn is seen with some context.
Held-out tasks (eval/heldout_tasks.json) are excluded.

  python -m distill.window --runs runs/teacher --out data/sft/train_resolved.jsonl --cap 8000 [--include-unresolved]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from distill.to_sft import episode_from_run  # noqa: E402


def groups(messages):
    """[(assistant_msg, [tool_msgs...]), ...] after the 2 prefix messages."""
    out, i = [], 2
    while i < len(messages):
        a = messages[i]; i += 1
        t = []
        while i < len(messages) and messages[i]["role"] == "tool":
            t.append(messages[i]); i += 1
        out.append([a] + t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/teacher")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tok", default="data/models/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--tools", default="distill/tools_schema.json")
    ap.add_argument("--system-prompt", default="submission/prompts/system.md")
    ap.add_argument("--cap", type=int, default=8000)
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--include-unresolved", action="store_true")
    ap.add_argument("--heldout", default="eval/heldout_tasks.json")
    ap.add_argument("--episodes-in", default=None, help="episodes.jsonl (e.g. from distill.from_atif) instead of --runs dirs")
    ap.add_argument("--tasks-file", default=None, help="JSON list; keep only these task ids (e.g. empty-patch-sound tasks)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tok)
    tools = json.load(open(args.tools))
    held = set(json.load(open(args.heldout))) if Path(args.heldout).exists() else set()
    sysp = Path(args.system_prompt).read_text()
    keep = set(json.load(open(args.tasks_file))) if args.tasks_file else None

    def ntok(msgs):
        t = tok.apply_chat_template(msgs, tools=tools, tokenize=False, preserve_thinking=True)
        return len(tok(t, add_special_tokens=False)["input_ids"])

    n_ep = n_win = n_whole = n_skipped_held = 0
    with open(args.out, "w") as f:
        if args.episodes_in:
            source = (json.loads(l) for l in open(args.episodes_in) if l.strip())
        else:
            source = (episode_from_run(d, sysp) for d in sorted(Path(args.runs).glob("*/s*")))
        for ep in source:
            if ep is None:
                continue
            if ep["task_id"] in held:
                n_skipped_held += 1
                continue
            if keep is not None and ep["task_id"] not in keep:
                continue
            if not ep["resolved"] and not args.include_unresolved:
                continue
            n_ep += 1
            msgs = ep["messages"]; prefix = msgs[:2]; gs = groups(msgs)
            base = ntok(prefix + gs[0][:1])  # cost floor with first assistant turn
            costs = []
            prev = ntok(prefix)
            for g in gs:  # incremental cost per group (additive estimate)
                cur = ntok(prefix + [m for x in gs[: len(costs) + 1] for m in x])
                costs.append(cur - prev); prev = cur
            P = ntok(prefix)
            total = P + sum(costs)
            if total <= args.cap:
                wins = [(0, len(gs))]; n_whole += 1
            else:
                wins, s = [], 0
                while s < len(gs):
                    e, used = s, P
                    while e < len(gs) and used + costs[e] <= args.cap:
                        used += costs[e]; e += 1
                    if e == s:  # single oversized turn: skip it
                        s += 1; continue
                    wins.append((s, e))
                    if e >= len(gs): break
                    s = max(s + 1, e - args.overlap)
            for k, (s, e) in enumerate(wins):
                wm = prefix + [m for g in gs[s:e] for m in g]
                n_win += 1
                f.write(json.dumps({"task_id": ep["task_id"], "sample": ep.get("sample", 0), "resolved": ep["resolved"],
                                    "window": k, "n_windows": len(wins), "turn_range": [s, e],
                                    "messages": wm}, ensure_ascii=False) + "\n")
    print(f"episodes used={n_ep} (whole={n_whole}) windows written={n_win} skipped_heldout={n_skipped_held} → {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
