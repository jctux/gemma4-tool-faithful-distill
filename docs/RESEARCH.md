# Research notes — SWE-agent distillation on commodity NVIDIA hardware (2026-10-25)

Evidence labels: **[V]** verified by us in this repo/box, **[S]** from a cited source, **[H]** hypothesis (untested).

## 1. The "adapters ignored" puzzle — resolved [V]
- On `vllm/vllm-openai:v0.19.1` (Docker image) our LoRA adapters changed nothing (identical log-probs to 6 decimals, even after 100× scaling of B), and `Gemma4ForConditionalGeneration` raised "does not support LoRA yet".
- The Kaggle wheelhouse wheel `vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl` (what the scorer runs) has **different files** from the Docker image with the same version string: md5 of `lora/model_manager.py`, `gemma4_mm.py`, `gemma4.py` all differ, and the wheel's `Gemma4ForConditionalGeneration` declares `SupportsLoRA` (image's does not).
- Serving E2B W4A16 + our adapters from that exact wheel (venv `~/vllm-env`, launcher `~/serve_wheel.sh`, GPU 1, no overrides): same prompt, 80 tokens, greedy → base Σlogprob −24.94; `lora_v1` −5.48; `lora_nothink` −6.23; ×100 probe → garbage. **Adapters are applied, and they already make the model emit tool calls on turn 1.**
- Our adapter key names (`base_model.model.model.language_model.layers.N...`) equal the official sample adapter's. Unsloth's regex `target_modules` was replaced by an explicit list.
- Upstream context [S]: vLLM issue #41754 (open; Unsloth Gemma 4 adapters "ignored despite loading"), #39815 (alias double-registration `self_decoder`/`cross_decoder`, closed not-planned), PR #39816 (fix, unmerged at last comment), #40693 / #39246 (LoRA support request). Lesson: **always validate adapters on the scorer's own wheel, never a different build**; keep a log-prob effect check (adapter vs base) as a gate.
- Local build recipe: torch 2.10.0+cu128 from the PyTorch index; wheelhouse vllm wheel; then `--no-deps` pins `transformers==5.13.1 tokenizers==0.22.2 xgrammar==0.2.5.post1`, `huggingface-hub<2`. Works on driver 550 (CUDA minor-version compatibility).

## 2. What the literature says about small-model SWE distillation
| Source | Setup | Finding |
|---|---|---|
| SWE-Next, arXiv 2603.20691 [S] | Full-FT Qwen2.5-Coder 7B/14B, 3.7k Claude/GPT trajectories, 32k ctx, lr 1e-5, ≤4 epochs, 8×A100 | 7B 17.4 / 14B 30.0 SWE-bench Verified; "clean" vs "recovery" (earlier failing test, then fix) buckets, but no per-bucket result; no data-quantity ablation |
| SWE-smith [S, via SWE-Next table] | 2k trajectories, 7B | 15.2 Verified |
| SWE-Gym [S] | 491 trajectories, 32B full-FT | +13.6 pts → 20.6 |
| Open-SWE-Traces, arXiv 2606.16038 [S] | 207k trajectories, Qwen3-30B-A3B family, 131k ctx | Training on **all** trajectories beats resolved-only (Verified 58.1 vs 55.3; Multilingual 47.6 vs 40.5): "excluding unresolved is counterproductive" |
| SWE-Prime, 2608.27449 [S, abstract only] | data selection | 10 % selected subset beats the full resolved set (model sizes not read) |
| Terminal-capability data engineering, 2602.21193 [S] | | a nontrivial share of trajectories exceed 32k and get truncated; extending context slightly hurt; long tail is noisy |
| Long-context multi-turn RL, 2508.03501 [S] | RL | dropping over-length trajectories can bias training: the dropped ones are often looping failures |
| LoRA Without Regret (Thinking Machines) [S, secondary summaries] | | LoRA ≈ full FT for RL and small-data SFT if MLP layers are adapted (attention-only underperforms); best LoRA lr ≈ 10× full-FT lr |
| Multi-turn GRPO, GTPO (ACL 2026), 2604.02869 [S] | | outcome-only reward gives every turn the same advantage; turn-level credit helps ~4 pts on program synthesis; dense rewards can hurt |
| DeepSWE / Ornith / Devstral roundups (Modal) [S] | | RL results used 64×H100 for 6 days (DeepSWE) or an 80 GB GPU (Ornith); nothing found for QLoRA-GRPO SWE agents on one 24 GB card |
| Community, this competition [S] | public LoRA attempt (r16, 30 steps, 412 turns) | lower loss but fewer solved runs (2/32 vs 4/32); single-agent configs beat analyzer pipelines in their runs |

**Gaps (no source found):** a LoRA-specific SWE-bench result at ≤4B effective parameters; chunked-CE memory numbers for a 262k-vocab model on 12 GB; any QLoRA-GRPO SWE-agent recipe on consumer GPUs.

## 3. Implications for our setup (commodity 2×RTX 3060 12 GB, driver 550)
1. **Eval locally is now possible and faithful [V]** (scorer wheel on a 3060). Use it as the gate: base vs adapter on held-out tasks, ≥50 runs, report resolved/N and s/task (12 h cap ⇒ ≈282 s/task mean [S, community calc]).
2. **Do not discard unresolved trajectories blindly [S: Open-SWE-Traces]**; run it as an ablation (resolved-only vs resolved+recovery vs all). Loss on failed episodes is a hypothesis for small LoRA data [H].
3. **Do not drop long episodes; shorten or segment them [S: 2508.03501, 2602.21193].** We currently lose 11 of 27 resolved episodes at the 8192-token cap [V]. Options: tell the teacher to avoid big outputs (cap reads/greps), split an episode at assistant-turn boundaries into overlapping windows with the system+task prefix, or buy memory with Unsloth chunked-loss/offloaded checkpointing (documented for Mistral-7B; unverified for 262k vocab [H]).
4. **Hyperparameters:** MLP+attention LoRA already; try lr 1e-4–5e-4 and more epochs — our 8–10 optimizer steps are far below what the literature trains for (hundreds of steps) [S, V].
5. **RL is out of reach on this hardware [S]**; the commodity-feasible substitutes are rejection-sampling fine-tuning with the student's own verified trajectories (expert iteration) and best-of-n with test-based selection [H].
6. **Data volume is the main lever [S]:** published SFT recipes use 0.5k (32B) to 3.7k (7B/14B) trajectories; we have 27 resolved. Prioritise teacher throughput (more samples per task, more tasks via soundness-audited extra repos).

## 4. Unverified claims to test next (in order)
1. Eval protocol sanity: run base E2B through `swegemma eval` on 20 held-out rich tasks against the local wheel server.
2. Adapter v1 vs nothink vs base on the same tasks (n≥50 runs).
3. Whether harness context compaction (`token_threshold=14336`) actually changes what the student sees [S conflicting: harness README says yes; a community repo says ADK never compacts within a task] — inspect a trace.
4. Windowed long-episode training vs dropping.
