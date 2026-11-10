"""Build notebook/kagle_distill_pitfalls.ipynb (no nbformat needed). Run: python notebook/build_nb.py"""
import json
from pathlib import Path

cells = []
def md(s): cells.append({"id": f"c{len(cells):02d}", "cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"id": f"c{len(cells):02d}", "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# Distilling a coding agent into Gemma 4 on commodity GPUs: results and pitfalls

Companion notebook to the paper-track writeup. CPU only, no internet, no model weights: it re-creates the paper's
tables and demonstrates the three pieces of our pipeline that are easy to get silently wrong
(empty-patch soundness control, windowing of long episodes, masking of tool output).

Everything here reads the small files in the `data/` folder of this notebook's dataset
(`results.json`, `sound_train_tasks.json`, `heldout_fastapi.json`).
""")

md("## 1. Held-out evaluation: 12 `fastapi` tasks, repeated runs")
code("""
import json, math, os
from pathlib import Path

import glob
cands = [Path("data"), Path("notebook/data")] + [Path(x).parent for x in glob.glob("/kaggle/input/**/results.json", recursive=True)]
DATA = next(p for p in cands if (p / "results.json").exists())
results = json.load(open(DATA / "results.json"))

def group(name):
    for pre, label in [("e12b_v2", "12B v2 (60 calls / 12 min + early-submit prompt)"), ("e12b_base", "12B base"),
                       ("e4b_prompt2", "E4B base + prompt2"), ("e4b_v3c", "E4B + Claude-teacher LoRA (v3c)"),
                       ("e4b_onpol", "E4B + on-policy LoRA (onpol)"), ("e4b_base", "E4B base")]:
        if name.startswith(pre):
            return label
    return name

rows = {}
for name, r in results.items():
    rows.setdefault(group(name), []).append(r)

print(f"{'config':58s} runs  resolved      patches        timeouts       mean calls")
for label, rs in rows.items():
    f = lambda k: " / ".join(str(r[k]) for r in rs)
    print(f"{label:58s} {len(rs):<5d} {f('resolved'):13s} {f('patches'):14s} {f('timeouts'):14s} {f('mean_calls')}")
""")

md("""
Each run is 12 held-out tasks through the official harness. Two things stand out: the adapters submit far fewer patches than
the base model, and none of the variants resolves clearly more than the base. Hidden-task scores from the Kaggle leaderboard:
E4B base **0.05**, E4B + prompt2 **0.03**.
""")
code("""
try:
    import matplotlib.pyplot as plt
    labels = list(rows)
    tot_res = [sum(r["resolved"] for r in rs) for rs in rows.values()]
    tot_n = [12 * len(rs) for rs in rows.values()]
    pat = [sum(r["patches"] for r in rs) / (12 * len(rs)) for rs in rows.values()]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    ax[0].barh(labels, [a / b for a, b in zip(tot_res, tot_n)]); ax[0].set_title("fraction resolved")
    ax[1].barh(labels, pat); ax[1].set_title("fraction of tasks with a submitted patch")
    plt.tight_layout(); plt.show()
except ImportError:
    print("matplotlib not available; skipping plot")
""")

md("""
### How noisy is a 12-task evaluation?

The same E4B base configuration resolved 2, 0 and 0 tasks in three runs. With a true resolve rate near 5.6 % per trial,
the chance that a 12-task run shows zero resolves, or two, is large:
""")
code("""
def binom_pmf(k, n, p): return math.comb(n, k) * p**k * (1 - p)**(n - k)

p = 2 / 36   # pooled E4B base rate
for k in range(0, 4):
    print(f"P(k={k} resolved of 12 | p={p:.3f}) = {binom_pmf(k, 12, p):.3f}")

# Smallest true-rate difference we could reliably see with 36 vs 36 trials (rough 80 % power, two-sided 5 %):
n = 36
for p0 in (0.056, 0.10):
    delta = 2.8 * math.sqrt(2 * p0 * (1 - p0) / n)
    print(f"base rate {p0:.3f}: need a difference of about {delta:.2f} (= {delta*n:.0f} extra resolves of {n}) to detect it")
""")

md("""
## 2. Pitfall: tasks that pass with an empty patch

A SWE task is only informative if it **fails** when the agent changes nothing. We applied an empty patch to each task in a fresh
container and kept only tasks whose hidden tests fail. In our environment 27 of 30 tested `rich` tasks passed with no fix,
so training on "resolved" `rich` trajectories would have taught the model to do nothing. Only the `fastapi` tasks below stayed sound.
""")
code("""
sound_train = json.load(open(DATA / "sound_train_tasks.json"))
heldout = json.load(open(DATA / "heldout_fastapi.json"))
print(len(sound_train), "sound training tasks;", len(heldout), "held-out tasks;", "overlap:", set(sound_train) & set(heldout))

def keep_if_sound(task_ids, empty_patch_passes):
    \"\"\"empty_patch_passes: dict task_id -> bool (hidden tests pass with NO change). Keep tasks that fail with it.\"\"\"
    return [t for t in task_ids if not empty_patch_passes.get(t, True)]   # unknown -> treat as unsound

# toy example
demo = {"a": False, "b": True, "c": False}
print(keep_if_sound(["a", "b", "c", "d"], demo))
""")

md("""
## 3. Pitfall: dropping long episodes loses the best data

At an 8k-token cap on a 12 GB GPU, dropping over-long episodes threw away 11 of 27 resolved trajectories. Instead we split each
episode into windows of whole assistant/tool turns that fit the cap and overlap by two turns. This is the same planning logic as
`distill/window.py`, with the tokenizer replaced by an injected per-turn cost.
""")
code("""
def plan_windows(prefix_cost, costs, cap, overlap=2):
    \"\"\"costs[i] = token cost of turn-group i. Returns list of (start, end) group ranges.\"\"\"
    if prefix_cost + sum(costs) <= cap:
        return [(0, len(costs))]
    wins, s = [], 0
    while s < len(costs):
        e, used = s, prefix_cost
        while e < len(costs) and used + costs[e] <= cap:
            used += costs[e]; e += 1
        if e == s:            # a single turn larger than the cap: skip it
            s += 1; continue
        wins.append((s, e))
        if e >= len(costs): break
        s = max(s + 1, e - overlap)
    return wins

costs = [900, 1200, 700, 2500, 800, 600, 1500, 400, 900, 1100]    # toy turn costs
print("whole episode:", sum(costs) + 600, "tokens; cap 4000")
for k, (s, e) in enumerate(plan_windows(600, costs, 4000)):
    print(f"window {k}: turns {s}..{e-1}  ({600 + sum(costs[s:e])} tokens)")
""")

md("""
## 4. Pitfall: the chat template puts tool output inside the model turn

Gemma 4's template renders tool results as `<|tool_response>...<tool_response|>` **inside** the model turn, so
response-only loss masking leaves tool output trainable (the model learns to hallucinate tool results). We mask those spans explicitly.
The demo works on a rendered string; the real code (`train/train_qlora.py`) does the same on token ids.
""")
code("""
import re

rendered = (
    "<|turn>user\\nFix the bug in utils.py<turn|>\\n"
    "<|turn>model\\n<|tool_call>call:read_file{path:<|\\"|>utils.py<|\\"|>}<tool_call|>"
    "<|tool_response>response:read_file{value:<|\\"|>def f(x): return x+1  # 480 more lines of file text ...<|\\"|>}<tool_response|>"
    "<|tool_call>call:submit_patch{}<tool_call|><turn|>\\n"
)

def trainable_mask(text):
    \"\"\"1 for characters the loss may see: inside a model turn, outside tool_response spans.\"\"\"
    mask = [0] * len(text)
    for m in re.finditer(r"<\\|turn>model\\n(.*?)<turn\\|>", text, flags=re.S):
        for i in range(m.start(1), m.end(1)):
            mask[i] = 1
    for m in re.finditer(r"<\\|tool_response>.*?<tool_response\\|>", text, flags=re.S):
        for i in range(m.start(), m.end()):
            mask[i] = 0
    return mask

mk = trainable_mask(rendered)
shown = "".join(c for c, k in zip(rendered, mk) if k)
assert "<|tool_response>" not in shown and "480 more lines" not in shown
print(f"trainable characters: {sum(mk)}/{len(rendered)}")
print(shown)
""")

md("""
## 5. Where do E4B base failures come from?

Over 3 runs x 12 tasks (36 trials) of the base model: 2 resolved, 29 submitted a patch that failed the hidden tests, 5 submitted no patch.
Of the 29 failed patches, 15 end in a pytest **collection error** (exit code 2): the hidden tests import a module or name that the
patch does not provide, so about half of the failures are interface mismatches rather than wrong logic. Our `prompt2` variant
targeted exactly this and did not move the score (local 2/36, hidden 0.03 vs 0.05).
""")
code("""
taxonomy = {"resolved": 2, "patch, tests fail (assertions)": 14, "patch, collection error (exit 2)": 15, "no patch": 5}
total = sum(taxonomy.values())
for k, v in taxonomy.items():
    print(f"{k:34s} {v:3d}  {100*v/total:5.1f} %")
""")

md("""
## 6. Reproducing the full pipeline

The full code (teacher bridge, windowing, QLoRA, evaluation drivers) lives in the project repository, linked from the writeup. In outline:

1. `teacher/`: Claude Code drives the real harness tools through an MCP bridge inside the competition sandbox; a fresh second container verifies the patch.
2. Keep only resolved runs on sound tasks (section 2); window episodes (section 3).
3. `train/train_qlora.py`: Unsloth QLoRA with tool-output masking (section 4), 4k-token windows on a Kaggle T4x2.
4. Serve base + LoRA from the **scorer's** vLLM wheel (the Docker image of the same version silently ignored adapters) and run the harness on the 12 held-out tasks, three or more times.

**Caveats.** Twelve tasks, one seed per training run, and a hidden score of 0.05 for the base model: our contribution is the negative
result and the checks above, not a better agent.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("kagle_distill_pitfalls.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
