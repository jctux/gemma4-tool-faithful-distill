#!/usr/bin/env bash
# Serve E2B W4A16 with the scorer-wheel vLLM + LoRA adapters (GPU 1), flags mirroring the scorer's setup_vllm_server.
cd ~/vllm-env
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 exec .venv/bin/vllm serve ~/models/gemma-4-E2B-it-qat-w4a16-ct \
  --served-model-name gemma-4-e2b-it-qat-w4a16-ct --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --reasoning-config '{"reasoning_start_str":"<|channel>","reasoning_end_str":"<channel|>"}' \
  --limit-mm-per-prompt '{"image":0,"audio":0}' --enable-lora --max-loras 3 --max-lora-rank 16 \
  --lora-modules lora_v1=$HOME/kagle/adapters_wheel/lora_v1 lora_nothink=$HOME/kagle/adapters_wheel/lora_nothink \
  --max-num-batched-tokens 4096 --max-num-seqs 4 --port 8000 "$@"
