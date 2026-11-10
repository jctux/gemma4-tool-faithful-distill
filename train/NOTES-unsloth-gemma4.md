# Unsloth × Gemma 4 — training notes (from unsloth.ai/docs/models/gemma-4/train, read 2026-10-25)

## Model ids / VRAM
- `unsloth/gemma-4-E2B-it` — LoRA 8–10 GB. Fits one RTX 3060 (12 GB).
- E4B — docs conflict: intro says 10 GB, LoRA section says 17 GB. Treat E4B QLoRA on 12 GB as *try, expect OOM at long seq*; fallback E2B or Kaggle L4x4.
- Load with `load_in_4bit=True` (QLoRA) or `load_in_16bit=True`.
- Use `FastModel` for text SFT (not FastVisionModel). Vision/audio layers off.

## Hyperparams (text SFT baseline)
- r=8..32, lora_alpha=r, dropout=0, bias="none", attention+MLP on, vision off.
- lr 2e-4 (2e-5 for long runs), batch 1, grad_accum 4, adamw_8bit, wd 0.001, seed 3407.
- max_seq_length 8192 (agent trajectories may need 16k → memory risk).
- warmup 5 steps, linear sched; `num_train_epochs=1`, `max_steps` off for real runs.

## Chat template
- Small models: template `gemma-4` (non-thinking). Format `<|turn>user\n...<turn|>\n<|turn>model\n...<turn|>`.
- Strip `<bos>` from templated text (processor adds it).
- `train_on_responses_only`: instruction_part `<|turn>user\n`, response_part `<|turn>model\n`.
- Tool-calling format NOT in Unsloth docs → get from HARNESS_README / Gemma 4 tokenizer chat_template.json.

## Gotchas
- Loss 13–15 on E2B/E4B start is normal.
- E2B/E4B produce garbage with `use_cache=False`; gradient checkpointing forces it → apply Unsloth's documented fix.
- Same template + EOS at inference as training, else garbage in other runtimes.

## Export
- Adapter only: `model.save_pretrained("adapters/main_lora")` (PEFT dir: adapter_config.json + adapter_model.safetensors) — competition needs this, not GGUF.
- Check `base_model_name_or_path` in adapter_config.json vs competition base `gemma-4-e4b-it-qat-w4a16-ct` (may need rewrite; verify in HARNESS_README).
