# Plan: Gemma 4 Developer Agent — Claude-distilled E4B/E2B LoRA (main + paper track)

## Context

Kaggle "Google - The Gemma 4 Developer Agent Competition" (deadline **Dec 2 2026**, $65k + $10k small-model bonus) and its **Paper Track** (deadline **Nov 12 2026**, $35k). Task: SWE-bench-style Python bug fixes (fastapi, rich, requests, httpx public train set; ~120 hidden test tasks from *private* repos). Submission = `submission.zip` with ADK `agent.yaml` + prompts + optional PEFT LoRA adapters. Base model must be one of `gemma-4-{31b,12b,e4b,e2b}-it-qat-w4a16-ct`; no external models at eval time (offline sandbox, 12h budget for all tasks).

Decisions made by jc (2026-10-25):
- Target **both** tracks.
- Train on **the GPU host**, 2× RTX 3060 12GB, by freeing one GPU. Unsloth docs: Gemma 4 E2B LoRA fits 8–10GB; E4B QLoRA recommended if it fits. Start E4B QLoRA, fall back to E2B. Kaggle L4x4 (96GB, free, competition-only) is the scale-up path.
- Teacher = **Claude Code CLI headless** (`claude -p`, jc's subscription), not raw API.
- jc already has a Kaggle account.

Method (paper core): **trajectory distillation**. Claude Code solves the 129 public tasks inside the competition sandbox using *exactly* the harness tool set (incl. `get_code_neighbors`, `search_similar_code`, `get_code_subgraph`). Trajectories verified with `test_patch`; passing ones → SFT data in Gemma 4 chat/tool-call format → QLoRA. Paper angles: (1) distillation under constrained tool schema, (2) ablation: graph tools vs plain file tools, (3) optional "new resource": extra tasks mined from the same 4 repos using the documented pipeline (fail-to-pass / pass-to-pass).

## Things only jc can do (blocking, do first)

1. Log into kaggle.com → **Join** both competitions and accept rules (`/competitions/gemma-4-developer-agent`, `/competitions/gemma-4-developer-agent-paper`). Data is invisible until joined.
2. Create API token at kaggle.com/settings → place at `~/.kaggle/kaggle.json` (chmod 600) on the Mac **and** on deb-gpu (`~jc/.kaggle/kaggle.json`). I won't handle the token.
3. Confirm which ollama to stop during training windows: `academy-ollama` (GPU1, 10GB) or `platform-ollama` (GPU0, 8GB + 1.8GB python). Plan assumes **academy-ollama / GPU1**.

## Phase 0 — Setup (Oct 25–27)

Local repo the project root (empty now) → `git init`, layout:

```
kagle/
  README.md
  pyproject.toml            # uv-managed, py3.11+
  data/                     # gitignored; real data lives on deb-gpu /mnt/nvme/kaggle/gemma4
  harness/                  # notes + wrappers around swegemma / adk-eval-core from HARNESS_README
  teacher/                  # Claude Code headless runner + MCP server exposing harness tools
  distill/                  # trajectory -> Gemma 4 SFT dataset conversion, filtering, stats
  train/                    # unsloth QLoRA script, configs
  eval/                     # local scoring on public tasks, ablation configs
  submission/               # agent.yaml, prompts/, adapters/ -> zip
  paper/                    # writeup.md (<=3000 words), figures, notebook
  scripts/                  # deb-gpu helpers: free_gpu.sh, sync.sh
```

Steps:
- Install `kaggle` CLI (uv tool) on Mac + deb-gpu. On deb-gpu (Python 3.13, no uv): install uv to `~/.local/bin`.
- `kaggle competitions download -c gemma-4-developer-agent -p /mnt/nvme/kaggle/gemma4` on deb-gpu (22 GB; 77 GB free). Also pull `HARNESS_README.md` + `sample_submission/` to Mac for reading.
- Read `HARNESS_README.md`; record: local eval CLI, sandbox lifecycle (two containers), tool signatures, context compaction, Gemma 4 chat/tool-call template expected by harness, adapter loading. **Open question to resolve here:** harness likely assumes Docker; on deb-gpu we use rootless Podman (global rules apply: no `sudo podman`, user jc only). Try `podman-docker` compat or `DOCKER_HOST=unix:///run/user/1000/podman/podman.sock`; if harness fights podman, run eval on Mac with Docker Desktop and train only on deb-gpu.
- Build `swebench-sandbox:latest` from `docker/Dockerfile.public`.
- `scripts/free_gpu.sh`: `podman stop academy-ollama` + verify `nvidia-smi` GPU1 free; `scripts/restore_gpu.sh`: `podman start academy-ollama`. Document in README. Never touch system-level units.
- **Baseline**: run `sample_submission` with vanilla `gemma-4-e4b-it-qat-w4a16-ct` on the 129 public tasks locally (or on a Kaggle notebook if local inference of the harness is impractical). Record pass rate = baseline number for paper.

## Phase 1 — Teacher trajectories (Oct 28–Nov 3)

- `teacher/mcp_tools.py`: MCP server (stdio) exposing the 9 harness tools with identical names/signatures: `run_command`, `submit_patch`, `get_status`, `read_file`, `edit_file`, `write_file`, `get_code_neighbors`, `search_similar_code`, `get_code_subgraph`. Graph tools read `graphs/<id>.json` (NetworkX JSON) and `embeddings/<id>.npz` (256-d, cosine). File/shell tools exec inside the task's sandbox container (`podman exec`). Logs every call/result as JSONL.
- `teacher/run_task.py`: per task → start sandbox from `snapshots/<id>.tgz`, run `sandbox/setup.py`, launch `claude -p` with system prompt = the one we will ship to Gemma (so the student sees the same prompt), tools restricted to the MCP server only (`--allowedTools mcp__harness__*`, disallow built-ins), problem_statement as user turn. Capture patch via `submit_patch`.
- `teacher/verify.py`: apply `test_patch`, run pytest → PASS/FAIL. Keep passing trajectories; also keep failing ones tagged (useful for DPO/rejection ablation later).
- Run N=2–3 samples per task (temperature via different seeds/prompts) across the 129 public tasks. Expect ~300 runs; rate-limit aware (serial or 2-wide), overnight.
- **Optional (new-resource prize + more data):** `distill/mine_tasks.py` replicate the dataset pipeline on the same 4 repos (commits touching `.py` + `test_*.py`, linked issue, fail-to-pass/pass-to-pass check) to add 100–300 more training tasks. Decide go/no-go on Oct 30 based on teacher throughput.

## Phase 2 — Distill + train (Nov 1–9)

- `distill/to_sft.py`: trajectory JSONL → Gemma 4 chat template with native tool-call format (get exact format from HARNESS_README / Gemma 4 tokenizer chat template; the harness must parse the student's calls). Truncate/compact long tool outputs the same way the harness does (context compaction). Produce train/val split by repo (hold out one repo for generalization check, since hidden test = private repos).
- `train/train_qlora.py` (Unsloth, per unsloth.ai Gemma 4 guide): base `unsloth/gemma-4-E4B-it` 4-bit, LoRA r=16–32 on attn+MLP projections, seq len 8–16k with gradient checkpointing, batch 1 + grad accum, 2–3 epochs. Run on deb-gpu GPU1 (`CUDA_VISIBLE_DEVICES=1`). If OOM → E2B (also qualifies for $10k small-model bonus either way, E4B and E2B both eligible).
- Export PEFT dir: `adapter_config.json` + `adapter_model.safetensors` → `submission/adapters/main_lora/`. Confirm adapter targets/names match the competition's `-qat-w4a16-ct` variant naming (check HARNESS_README; may need `base_model_name_or_path` rewrite).
- `eval/score.py`: local pass rate on held-out repo + full public set. Compare: baseline vs LoRA; graph tools on vs off (ablation for paper).
- First Kaggle submission by **Nov 6** (zip via `kaggle competitions submit`). Check public LB.

## Phase 3 — Paper (Nov 8–12, hard deadline Nov 12 23:59 UTC)

- `paper/writeup.md` ≤3000 words: Title/subtitle, Abstract, Intro, Method (teacher setup, tool parity, filtering, SFT format, QLoRA), Experiments (baseline vs distilled, E4B vs E2B, graph-tool ablation, held-out repo), Related work (SWE-bench, SWE-smith/SWE-Gym style distillation, LoRA/QLoRA, Gemma 3n/4 PLE), Limitations, Citations.
- Public Kaggle notebook with eval + data-conversion code; link in Writeup. Code Apache-2.0.
- jc submits Writeup on kaggle.com (I draft; submission click is jc's).

## Phase 4 — Leaderboard iteration (Nov 13–Dec 2)

- Round 2: student self-distillation / rejection sampling on public tasks; DPO on pass/fail pairs if time.
- Prompt + sub-agent config tuning in `agent.yaml` (analyzer sub-agent with read-only tools, `tool_lora` adapter).
- E2B variant as second final submission (two finals allowed).
- Scale-up run on Kaggle L4x4 if local is the bottleneck.

## Risks

- **Hidden test = private repos.** Distilling only on 4 public repos risks overfit; mitigate via held-out-repo eval + mined extra tasks + generic tool-use behaviour over repo-specific knowledge.
- **12GB VRAM.** E4B QLoRA at long contexts may OOM → E2B or Kaggle L4x4.
- **Docker vs Podman** harness friction on deb-gpu.
- **Claude Code rate limits** on ~300 agent runs; pace overnight, 2-wide max.
- **Prod impact**: stopping an ollama affects whichever stack uses it; only during training windows, restore after.
- Nov 12 is 18 days out; paper needs at least one trained adapter + eval numbers by ~Nov 8.

## Verification

- Phase 0: `kaggle competitions files -c gemma-4-developer-agent` lists 524 files; sandbox image builds; baseline run writes `submission.parquet` with 129 rows and a pass rate.
- Phase 1: `teacher/verify.py` reports pass/fail per trajectory; ≥60 passing trajectories.
- Phase 2: training loss decreases; local pass rate on held-out repo ≥ baseline; `kaggle competitions submit` succeeds and LB score appears.
- Phase 3: Writeup saved + submitted before Nov 12 23:59 UTC; notebook public.

## Amendment 2026-10-25 (after Phase 0)
- deb-gpu: CPU-only duty (sandboxes for teacher runs + verification). No ollama changes, no driver upgrade (vLLM 0.19.1 needs CUDA 12.9 / driver ≥575; box has 550).
- Training (QLoRA) and evaluation (official harness + vLLM) both run on **Kaggle L4x4 notebooks**. Adapters/episodes move as Kaggle Datasets.
- Verifiable public pool so far: rich (gold passes), fastapi only with pin-aware dependency resolution (local patch in `teacher/harness_env.py`); requests/httpx unverifiable.
