# swegemma harness — key facts (from HARNESS_README.md, read 2026-10-25)

## Stack
- `adk-submission` compiles `agent.yaml` (+ sub_agents, prompts via `!include`, skills, adapters) into Google ADK agent tree. No Python entrypoints; declarative YAML only.
- `adk-eval-core`: sandbox backends (`DockerContainer`, `SubprocessSandbox`), 3-tier `apply_replacement` for edit_file, ATIF v1.7 trajectory tracing (`SessionTrace`).
- `swegemma`: two-container lifecycle, 9 tools, prompt builder, patch extraction (`scripts/inference.py`), pytest verify (`scripts/metric.py`).

## Eval hardware / serving (what the student must run under)
- 4× L4 (96 GB), vLLM tp=4, `max_model_len=32768`, `tool_call_parser=gemma4`, `reasoning_parser=gemma4`, reasoning delimiters `<|channel>` … `<channel|>`, tool call tag `<|tool_call>`.
- Allowed bases: `gemma-4-{31b,12b,e4b,e2b}-it-qat-w4a16-ct`. One base per submission. bf16 aliases (`gemma-4-e4b-it`, `gemma-4-e2b-it`, …) exist for **local CLI only**.
- LoRA: `adapters/<name>/{adapter_config.json,adapter_model.safetensors}`; `adapter: <name>` on any LlmAgent; max rank 128; total unpacked zip < 3 GiB; `.safetensors` only.
- `generate_content_config`: temperature/top_p/top_k/max_output_tokens (≤32768, default 16384)/seed/thinking_config. With vLLM + LoRA use `include_thoughts: true` + `thinking_budget: N` (not `thinking_level`). `thinking_budget: 0` disables thinking.

## Per-task budgets (defaults in inference.py; override via `eval_config.yaml` → `evaluation:`)
- time 60 min, tool_calls 100, turns 500, command timeout 300 s, stdout cap 5000 chars, read_file cap 150 lines / 10k chars.
- Sample submission uses 1 min / 10 calls / 50 turns (tiny!). Overall 12 h for all ~120 test tasks ⇒ ~6 min/task average. Budget per task matters.
- Container setup excluded from agent clock.
- Context compaction: ADK EventsCompaction interval 15, token_threshold 14336, retention 5.

## Prompt protocol (what teacher must replicate exactly)
- session.state: `problem_description`, `hints`. Instruction may template `{problem_description}`.
- Initial user msg = header + problem statement + hints + budget block + env rules + standard instructions 0–5 + code-intelligence tool blurb (only declared tools) + workspace layout (`find . -maxdepth 3`, ≤200 entries).
- Loop ends on `submit_patch()` + final text, timeout, budget, or 3 consecutive no-tool turns. Three nudge messages (token-limit cut tool call / MAX_TOKENS / plain "please continue").
- Tool errors returned as `{"status":"error","error_type":...}` not aborts.

## Tools (names + signatures must match in teacher MCP)
1. `run_command(command)` → `{status, stdout, stderr, exit_code}`; bash -c in /workspace; counts.
2. `submit_patch()` → `git add -N . && git diff --binary HEAD`; does NOT count.
3. `get_status()`; does NOT count.
4. `read_file(filepath, start_line?, end_line?)` 1-indexed inclusive; 150 lines/10k chars cap.
5. `edit_file(filepath, old_string, new_string, allow_multiple=False)` exact→flexible→regex; uniqueness enforced.
6. `write_file(filepath, content)`.
7. `get_code_neighbors(node, edge_type?, max_neighbors=50)` 4-tier name resolution.
8. `search_similar_code(query, k=10)` cosine over .npz by resolved node; lexical fallback.
9. `get_code_subgraph(nodes)`.
- Paths relative to /workspace; `..` rejected. All outputs JSON strings.

## Sandbox
- Docker backend: image `swebench-sandbox:latest`, `network none`, 4 GiB RAM, 2 vCPU. **Subprocess backend** (`--sandbox subprocess`) exists for hosts w/o Docker daemon → use on deb-gpu (rootless podman) or set DOCKER_HOST to podman socket.
- Bootstrap: wheels → snapshot extract → git exclude → `pip install -e` → test deps → pytest.ini + conftest.py → baseline commit. Don't touch pytest.ini/conftest.py.
- Verify: fresh container, apply agent patch (4-pass), reset test files, apply test_patch, pytest on target tests; exit 0 ⇒ resolved.

## Implications for our plan
- Teacher (Claude Code) MCP tools must emit identical JSON shapes & caps so student sees same distribution.
- Student SFT data must use Gemma 4 native tool-call syntax parsed by vLLM `gemma4` parser (get template from tokenizer `chat_template`). Thinking channel optional; consider training w/ short thoughts or none (`thinking_budget: 0`) to save tokens under 32k ctx.
- Per-task budget tuning in `eval_config.yaml` is a free lever (12 h total cap).
- Local baseline: run CLI with `--sandbox subprocess` + bf16/4-bit local model, or vLLM in a podman container on deb-gpu.

## Where the harness code lives (found 2026-10-25)
- NOT on PyPI. Kaggle dataset `metric/gemma-4-developer-agent-wheelhouse` (also what the scorer mounts): `swegemma-0.2.11`, `adk_submission-0.2.13`, `adk_eval_core-0.1.0`, `google_adk-1.36.1`, `vllm-0.19.1`, `transformers-5.13.1`, `bitsandbytes-0.46.1`, `compressed_tensors-0.15.0.1`, …
- Local copy: `data/wheelhouse/` (4 small wheels). Installed in `.venv`; `swegemma eval --help` works.
- Sample adapter config: `base_model_name_or_path: google/gemma-4-31b-it-qat-w4a16-ct` ⇒ ours should be `google/gemma-4-e4b-it-qat-w4a16-ct` (HF id pattern). Sample adapters are dummies (r=4, q/o_proj, layer 0).
- Sample submission uses 31B + 2 dummy LoRAs + `code_analyzer` AgentTool; budgets 1 min / 10 calls / 50 turns; sampling T=0.2, top_p 0.95, max_out 16384, thinking 4096 w/ include_thoughts.

## Public leaderboard snapshot (2026-10-25)
- Top = 0.24 (4 teams), huge cluster at 0.22 (~40 teams) → shared public notebook; 0.20 next. 3,748 teams. ~58 public test tasks ⇒ 0.24 ≈ 14 tasks.
- Community intel (happyc0der/gemma-swe-agent): 31B impossible on T4; ADK context overflow after ~20 big tool outputs; low temperature ⇒ verbatim retry loops; scratch files leak into patch; best LB they got 0.12.
- Implication: bar is low. E4B+LoRA may not top the main LB (31B rules there) but small-model prize ($10k, E2B/E4B only) is a separate ladder. Paper track judged on method not LB.

## Local CLI = OpenAI-compatible client (cli.py `_build_cli_model_registry`)
- Reads `MODEL_PROXY_URL` | `LITELLM_API_BASE` | `LOCAL_INFERENCE_URL` | `OPENAI_BASE_URL` (default `http://localhost:8000/v1`) + matching API key env (default `EMPTY`).
- Registers each declared model alias as `openai/<alias>` and each adapter name as `openai/<adapter_name>` → the server must expose model ids equal to the alias (e.g. `gemma-4-e4b-it-qat-w4a16-ct`) and LoRA ids equal to adapter dir names (vLLM `--lora-modules main_lora=/path`, `--served-model-name`).
- `--models-yaml` can map alias → `path` + `api_base` per model.
- CLI does NOT launch vLLM; we run it ourselves (podman `vllm/vllm-openai:v0.19.1`, same version as scorer wheelhouse).

## Models (HF, all ungated, verified 2026-10-25)
- `google/gemma-4-E4B-it-qat-w4a16-ct` 11.5 GB on disk (W4A16 transformer + bf16 PLE/vision/audio); `google/gemma-4-E2B-it-qat-w4a16-ct` 8.3 GB.
- bf16: `google/gemma-4-E4B-it` / `unsloth/gemma-4-E4B-it` 16 GB; E2B 10.25 GB.
- Kaggle Models also hosts the 4 ct variants (for notebooks). 3060 = sm_86 Ampere → vLLM INT4 kernels OK.
- Downloading E4B-ct to deb-gpu `/mnt/nvme/kaggle/models/`.

## Gold-patch self-check on deb-gpu (2026-10-25, podman via DOCKER_HOST socket)
- Full pipeline works: Container A bootstrap ~17 s, 9 tools OK, Container B verify OK.
- `rich_3061`: gold → resolved=True (112 passed).
- `requests_6592`: gold → 185 ERRORs: `recursive dependency involving fixture 'httpbin'` (repo conftest wraps pytest-httpbin's fixture; plugin not active in sandbox). Treat the 13 requests tasks as unverifiable locally.
- `httpx_3672` (only httpx task): gold → collection error `module 'httpx' has no attribute 'Stream'` (repo is the v1 "ahttpx" rewrite). Skip.
- ⇒ verifiable training pool = fastapi (67) + rich (48). Hold out rich (or a fastapi slice) for generalization checks.

## First teacher run (rich_3061, 2026-10-25 13:19)
- End-to-end OK: sandbox 27 s, Claude 94 s, 8 counted tool calls, patch 3.4 kB, verify 32 s.
- Not resolved: 111/112 pass, `test_panel.py::test_render_size` fails (gold fix also changes segment merging). Teacher solves the *described* problem; hidden tests may cover more ⇒ need multiple samples + only-resolved filtering.
- Claude leaned on `run_command grep/sed` for navigation rather than graph tools — prompt nudge toward `search_similar_code`/`get_code_neighbors` if we want the student to learn graph tools (ablation axis for the paper).
- Observed: `read_file` called with `/workspace/...` absolute path (harness accepts). Keep.
- Wheelhouse gap: pydantic 2.13.4 present but `typing_inspection` missing → fastapi gold fails at import. Added wheel locally (`data/gemma4/wheels`, pushed, image rebuilt).

## Local serving blocker (2026-10-25 15:50)
- `vllm/vllm-openai:v0.19.1` = torch 2.10 + CUDA 12.9 → needs NVIDIA driver ≥ 575. deb-gpu has 550.163 (Debian trixie stable; no newer package in configured repos). GeForce cards can't use cuda-compat forward compatibility → `Error 804: forward compatibility was attempted on non supported HW`.
- Training is unaffected (torch cu124 wheels work on driver 550). Only vLLM serving (local `swegemma eval`) is blocked.
- Options: (a) upgrade driver on deb-gpu (sudo, reboot, prod box) → user decision; (b) evaluate adapters on Kaggle notebooks (official L4x4 env, free, most faithful) — default plan; (c) llama.cpp server with GGUF + LoRA (CUDA 12.4 build) — extra conversion work.
- platform-ollama reloads models on demand (`keep_alive` forever); it re-grabbed GPU1 within minutes of `ollama stop`. To keep a GPU free for training, the container must be stopped (or pinned to GPU0 via CUDA_VISIBLE_DEVICES) — auto-mode blocks me from stopping it; jc must do it.

## Community intel: "Task Soundness, Serving and Evaluation" (dmitriigluzdov, Kaggle, read 2026-10-25)
- 114/129 public tasks "sound" (empty patch fails, gold passes) **after their dependency repairs** (`dependency_repairs.csv`); statuses: sound / passes_without_fix / control_error / gold_does_not_pass_local_grader. Confirms our fastapi issue is environment, not data.
- Single coder (no analyzer sub-agent) 27/70 dev vs reference coder+analyzer 16/70; on 44 rich tasks both 12/44 but single coder 216 s vs 381 s mean. ⇒ ship a single-agent config; budget ≈ 282 s/task mean to fit 12 h for 120 tasks with 15 min startup.
- thinking_budget 2048 → 50 resolved vs 4096 → 48 (public screen). Smaller thinking is fine.
- Their LoRA (r16, α32, lr 1e-4, 30 steps, 412 turns from 23 dev tasks, ≤8192 tok) lowered loss 20–30 % but solved fewer runs (2/32 vs 4/32 base). Lesson: tiny SFT sets hurt; we need hundreds of verified episodes + eval on ≥50 runs to see signal.
- Starter notebook (ryanholbrook) = the eval template: wheelhouse at `/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse`, models at `/kaggle/input/models/google/gemma-4/other/<alias>/2`, `VllmServer`/`VllmConfig` from adk_submission, tp = #GPUs, `machine_shape: NvidiaL4`, internet off.

## Pin-aware dependency fix — WORKS (2026-10-25 16:09)
- Root cause: `_ensure_container_site_packages` untars the newest wheel of every package into site-packages **without RECORD files**, so pip cannot uninstall/downgrade them (`Cannot uninstall starlette 1.6.0 … no RECORD`).
- Fix (`teacher/harness_env.py::install_test_dependencies_pinned`): after stock bootstrap, parse `[project].dependencies` + requirements*.txt, for each spec not already satisfied: purge the injected package dir + dist-info, `pip install --no-index --find-links=/wheels --ignore-installed --no-deps <spec>`; exact test-dep pins fall back to the bare name. Extra wheels (dirty-equals, sqlmodel, flask, PyJWT, pytest-httpbin, trio, …) added to the local wheelhouse (never pytest/rich/pydantic to avoid shadowing).
- Result: fastapi_12942 / 14077 / 11194 gold → resolved=True (~2.5–3 min each). Full sweep → `runs/gold/index_pinned.jsonl`.
- Caveat: local-only environment; the scorer's private repos use their own images. Student policy is unaffected (same tools/prompts).
