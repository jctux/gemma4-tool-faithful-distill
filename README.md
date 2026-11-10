# Tool-faithful trajectory distillation for a small Gemma 4 SWE agent

Code, task lists and run outputs for the Kaggle paper-track writeup
**"Tool-Faithful Distillation for a Small Gemma 4 SWE Agent: What Silently Failed"**
(Google - The Gemma 4 Developer Agent Competition, 2026).

- Writeup: https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/tool-faithful-distillation-for-a-small-gemma-4-swe
- Companion notebook (CPU, no internet): https://www.kaggle.com/code/tuxmx1/kagle-distill-pitfalls
- Data release (adapters, local and L4x4 eval outputs, on-policy episodes, task lists):
  https://www.kaggle.com/datasets/tuxmx1/gemma4-tool-faithful-distill
- Competition harness: Kaggle dataset `metric/gemma-4-developer-agent-wheelhouse` (`swegemma`), see its `HARNESS_README.md`.

## What is here

| path | what |
|---|---|
| `teacher/` | **Harness bridge.** `harness_env.py` reuses the official `swegemma` bootstrap, tool factory and prompt builder; `mcp_server.py` exposes the nine harness tools over MCP (streamable HTTP) so any tool-using agent can act *inside* the scoring sandbox; `run_task.py` / `batch.py` drive Claude Code headless as the teacher; `verify.py` runs the hidden tests in a fresh container; `empty_check.py` is the empty-patch soundness control; `gold_check.py` the gold-patch audit. |
| `distill/` | Trajectory -> SFT episodes in Gemma 4 chat/tool-call format (`to_sft.py`), tool schemas derived from the ADK definitions (`tools_schema.py`), windowing of long episodes (`window.py`). |
| `train/` | Unsloth QLoRA (`train_qlora.py`) with explicit masking of `<|tool_response>` spans; notes on Gemma 4 + Unsloth pitfalls. |
| `eval/` | Held-out task lists (`heldout_fastapi.json`, 12 tasks), sound training tasks (`sound_train_tasks.json`, 35 tasks), local eval driver (`run_eval.py`), submission builder (`make_sub.py`). |
| `kaggle/` | L4x4 eval and training kernels (`eval_kernel.py`, `train_kernel.py`) and their pushers (`push_eval.py`, `push_train.py`). |
| `submissions/` | Exact file sets of every hidden-task submission with their scores (`submissions/README.md`). |
| `resources/` | Gold-patch audit index with per-snapshot pins (`gold_index_pinned.jsonl`), teacher run index (`teacher_index.jsonl`: task, sample, resolved, tool calls, seconds), held-out results used in the paper (`heldout_results.json`), per-task results of the two 129-task 31B replicas on L4x4 (`replica_31b_*_task_results.jsonl`). |
| `notebook/` | Source of the companion notebook. |
| `paper/` | `writeup.md` (the paper), `make_fig.py` and `fig_hidden_scores.png`. |
| `scripts/` | Host helpers (vLLM serving from the scorer's wheel, data sync, sandbox cleanup). |
| `docs/` | Plan and research notes (`RESEARCH.md`: literature table, the vLLM adapter-effect investigation). |

## Reproducing the pipeline

1. **Sandbox + data.** Download the competition data; build `swebench-sandbox:latest` from the harness `docker/Dockerfile.public`. `teacher/harness_env.py` adds per-snapshot pin resolution from `/wheels` so older snapshots install (see `resources/gold_index_pinned.jsonl` for which tasks the gold patch resolves).
2. **Soundness control.** `python -m teacher.empty_check` applies an empty patch and runs the hidden tests; keep tasks that fail. Run tests as `python -m pytest`, not bare `pytest`: with bare `pytest` the installed `rich` wheel shadows the workspace and 27 of 30 `rich` tasks pass with no fix.
3. **Teacher runs.** `python -m teacher.mcp_server` next to the sandbox host, then `python -m teacher.batch --repo fastapi --samples 2`. Claude Code runs headless with built-in tools disabled, so every action goes through the harness tools and is logged; `verify.py` grades each run in a fresh container.
4. **Episodes.** `python -m distill.to_sft --runs runs/teacher --out data/sft/episodes.jsonl --only-resolved`, then window with `distill/window.py` (3.8k-8k tokens, overlap two turns).
5. **Training.** `train/train_qlora.py` (rank 16, attention + MLP, lr 1e-4 to 2e-4, 2-3 epochs); locally on a 12 GB GPU for E2B, on Kaggle for E4B via `kaggle/push_train.py`.
6. **Evaluation.** Serve base + LoRA from the **scorer's own vLLM wheel** (`scripts/serve_wheel.sh`): the Docker image with the same version string silently ignores Gemma 4 LoRA adapters. Then `eval/run_eval.py` runs the official harness on the 12 held-out tasks, three or more times. For 31B, `kaggle/push_eval.py` runs all 129 public tasks on an L4x4 notebook.

Environment variable `GPU_HOST` (default `user@gpu-host`) points the scripts at the machine that
hosts the sandboxes and the vLLM server.

## Main results (details and caveats in the writeup)

- Hidden public-split scores: E4B base 0.05, E4B + prompt variants 0.03-0.05, 31B base 0.13, 31B public prompts 0.12-0.13; four of seven 31B submissions with larger budgets hit the scorer's 12-hour "Notebook Timeout".
- On 12 held-out `fastapi` tasks (3 runs each), adapters trained on Claude trajectories or on the model's own resolved runs resolved 0 of 36 trials (base: 2 of 36) and submitted far fewer patches.
- Four silent failure modes documented: vLLM build ignoring LoRA without error, tool output left trainable under response-only masking, repetition loops from small-data LoRA, stale dataset uploads.

## License and credits

Apache License 2.0 (see `LICENSE`). The competition harness (`swegemma`) and the public prompts
reused in `submissions/` keep their own licenses and credits (see `submissions/README.md`).
Teacher trajectories were produced with Claude Code; the SFT episodes derived from them are not
part of the public data release.
