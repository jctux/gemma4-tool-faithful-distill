# Local QLoRA on deb-gpu (2× RTX 3060 12 GB) — set up 2026-10-25

- Env: `~/train/.venv` (python 3.12, root disk): torch 2.14.1+cu126 (works on driver 550), torchvision **from the cu126 index** (PyPI build breaks Unsloth's check), unsloth 2026.10.8, transformers 5.17, trl 1.15 (`processing_class=`, not `tokenizer=`), peft 0.21, bitsandbytes 0.50.
- Weights: `~/models/gemma-4-E2B-bnb4` (unsloth/gemma-4-E2B-it-unsloth-bnb-4bit, 7.6 GB). deb-gpu→HF is ~7 KB/s; fetch on the Mac with `scripts/hf_fetch.sh REPO DEST` (16 MB/s) and rsync over LAN.
- Smoke run (E2B, r16, 3 steps, 13 episodes ≤10k tokens): 53 s/step at grad_accum 2, loss ≈1.9, adapter 96 MB. 12k tokens OOMs on 12 GB; 10240 works with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Cost: ~25 s per 10k-token episode ⇒ 100 episodes × 2 epochs ≈ 1.5 h per GPU.
- Command: `CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 ~/train/.venv/bin/python train/train_qlora.py --episodes data/sft/episodes.jsonl --tools distill/tools_schema.json --base ~/models/gemma-4-E2B-bnb4 --out runs/train/<name> --r 16 --epochs 2 --max-seq-len 10240 --base-model-name google/gemma-4-E2B-it-qat-w4a16-ct`
- Adapter targets 7 projection types (k/v only on non-kv-shared layers); trainer now rewrites `target_modules` to a list for vLLM. Still unverified against vLLM (needs Kaggle L4 or a local vLLM).
- Idea to unblock local vLLM: the vllm image's CUDA *compat* libs caused Error 804 on driver 550; try running it with `-e LD_LIBRARY_PATH=/usr/local/nvidia/lib64` (no `/usr/local/cuda/compat`).
- 7/20 resolved episodes are >10k tokens and get dropped; context-length is the main data loss. Teacher prompts that avoid giant tool outputs would help.

## Local vLLM unblocked (2026-10-25 18:30)
- Error 804 cause: the vllm image's CDI-injected `/etc/ld.so.conf.d/00-compat-<uuid>.conf` pulls CUDA-575 compat libs ahead of the host driver 550. Deleting that file + `ldconfig` inside the container → `torch.cuda.is_available()` True on the 3060. `scripts/serve_vllm.sh` now does this.
- E2B W4A16 (8.3 GB) fits a 12 GB card for serving; E4B W4A16 (11.5 GB, PLE in bf16) does not.
