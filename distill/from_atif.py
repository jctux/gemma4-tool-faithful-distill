"""Convert on-policy eval traces (ATIF) into SFT episodes (same message schema as distill/to_sft.py).

Keeps only resolved runs that called submit_patch explicitly, so the student is not trained to stop without submitting.

  python -m distill.from_atif --evals runs/eval --prefix samp_ --tasks-file eval/sound_train_tasks.json --out data/sft/onpolicy_episodes.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def episode_from_trace(trace: dict, max_thought_chars: int = 1200) -> dict | None:
    steps = trace["steps"]
    msgs: list[dict] = []
    n_submit = n_calls = 0
    for s in steps:
        src = s["source"]
        if src == "system":
            msgs.append({"role": "system", "content": s["message"]})
        elif src == "user":
            msgs.append({"role": "user", "content": s["message"]})
        elif src == "agent":
            text = (s.get("message") or "")[:max_thought_chars]
            tcs = s.get("tool_calls") or []
            if not tcs:
                msgs.append({"role": "assistant", "content": text})
                continue
            calls = [{"id": tc["tool_call_id"], "type": "function",
                      "function": {"name": tc["function_name"], "arguments": tc.get("arguments") or {}}} for tc in tcs]
            msgs.append({"role": "assistant", "content": "", "reasoning_content": text, "tool_calls": calls})
            obs = s.get("observation")
            results = obs if isinstance(obs, list) else [obs] if obs else []
            for i, tc in enumerate(calls):
                r = results[i] if i < len(results) else None
                content = r.get("content") if isinstance(r, dict) else None
                if content is None:
                    return None  # missing tool result: drop the episode rather than invent one
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"], "content": content})
                n_calls += 1
                n_submit += tc["function"]["name"] == "submit_patch"
    if len(msgs) < 4 or msgs[-1]["role"] != "assistant":
        return None
    return {"messages": msgs, "n_tool_calls": n_calls, "n_submit": n_submit}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="runs/eval")
    ap.add_argument("--prefix", default="samp_")
    ap.add_argument("--tasks-file", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    keep = set(json.load(open(args.tasks_file))) if args.tasks_file else None
    n_res = n_nosubmit = n_bad = n_out = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for d in sorted(Path(args.evals).glob(f"{args.prefix}*")):
            res = d / "task_results.jsonl"
            if not res.exists():
                continue
            for line in res.read_text().splitlines():
                r = json.loads(line)
                if not r["resolved"] or (keep and r["instance_id"] not in keep):
                    continue
                n_res += 1
                tp = d / "traces" / f"trace_{r['instance_id']}.json"
                ep = episode_from_trace(json.loads(tp.read_text())) if tp.exists() else None
                if ep is None:
                    n_bad += 1; continue
                if ep["n_submit"] == 0:
                    n_nosubmit += 1; continue
                n_out += 1
                f.write(json.dumps({"task_id": r["instance_id"], "sample": d.name, "resolved": True, **ep}, ensure_ascii=False) + "\n")
    print(f"resolved={n_res} unusable={n_bad} no_submit_patch={n_nosubmit} written={n_out} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
