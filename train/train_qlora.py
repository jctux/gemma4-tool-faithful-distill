"""QLoRA SFT of Gemma 4 (E4B/E2B) on teacher episodes with Unsloth.

Input : data/sft/episodes.jsonl (from distill/to_sft.py) + distill/tools_schema.json
Output: PEFT adapter dir (adapter_config.json + adapter_model.safetensors) ready for submission/adapters/<name>/

  CUDA_VISIBLE_DEVICES=1 .venv/bin/python train/train_qlora.py --episodes data/sft/episodes.jsonl \
      --base unsloth/gemma-4-E4B-it --out runs/train/e4b_r16 --r 16 --epochs 2 --max-seq-len 16384
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset


def load_episodes(path: Path, tools: list[dict], holdout_repo: str | None, max_seq_len: int, tokenizer, preserve_thinking: bool = True):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    train, val = [], []
    n_long = 0
    for ep in rows:
        msgs = ep["messages"]
        if not preserve_thinking:  # ablation: drop teacher thoughts entirely
            msgs = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in msgs]
        text = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False,
                                             preserve_thinking=preserve_thinking)
        if text.startswith("<bos>"):
            text = text[len("<bos>"):]  # processor re-adds it
        n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tok > max_seq_len:
            n_long += 1
            continue
        rec = {"text": text, "task_id": ep["task_id"], "n_tok": n_tok}
        repo_short = ep["task_id"].rsplit("_", 1)[0]
        (val if holdout_repo and repo_short == holdout_repo else train).append(rec)
    print(f"episodes: {len(rows)} total, {len(train)} train, {len(val)} val, {n_long} dropped (> {max_seq_len} tokens)")
    return Dataset.from_list(train), (Dataset.from_list(val) if val else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="data/sft/episodes.jsonl")
    ap.add_argument("--tools", default="distill/tools_schema.json")
    ap.add_argument("--base", default="unsloth/gemma-4-E4B-it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-seq-len", type=int, default=24576)
    ap.add_argument("--no-thinking", action="store_true", help="drop teacher thoughts (ablation)")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--holdout-repo", default=None, help="e.g. 'rich' → validation split by repo")
    ap.add_argument("--load-in-4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap optimizer steps (smoke tests)")
    ap.add_argument("--base-model-name", default="google/gemma-4-E4B-it-qat-w4a16-ct",
                    help="value written to adapter_config.base_model_name_or_path for the harness")
    args = ap.parse_args()
    random.seed(args.seed)

    from unsloth import FastModel  # noqa: E402  (import after CUDA env is set)
    from unsloth.chat_templates import train_on_responses_only  # noqa: E402
    from trl import SFTConfig, SFTTrainer  # noqa: E402

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_len,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.r,
        lora_alpha=args.alpha or args.r,
        lora_dropout=0.0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
    )

    tools = json.loads(Path(args.tools).read_text(encoding="utf-8"))
    tok = getattr(tokenizer, "tokenizer", tokenizer)  # text tokenizer even when a processor is returned
    train_ds, val_ds = load_episodes(Path(args.episodes), tools, args.holdout_repo, args.max_seq_len, tok, preserve_thinking=not args.no_thinking)

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=str(Path(args.out) / "trainer"),
            report_to="none",
            max_length=args.max_seq_len,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            eval_strategy="epoch" if val_ds is not None else "no",
            save_strategy="epoch",
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )
    # Gemma 4 renders tool results INSIDE the model turn, so train_on_responses_only leaves them trainable.
    # Mask <|tool_response> ... <tool_response|> spans so the model is not trained to predict tool output.
    tr_s = tok.convert_tokens_to_ids("<|tool_response>")
    tr_e = tok.convert_tokens_to_ids("<tool_response|>")
    assert tr_s is not None and tr_e is not None and tr_s != tr_e and tr_s != tok.unk_token_id, (tr_s, tr_e)

    def _mask_tool_outputs(ex):
        labels, inside = list(ex["labels"]), False
        for i, t in enumerate(ex["input_ids"]):
            if t == tr_s:
                inside = True
            if inside:
                labels[i] = -100
            if t == tr_e:
                inside = False
        return {"labels": labels}

    def _stats(ds, tag):
        tot = sum(len(x) for x in ds["labels"]); keep = sum(sum(1 for y in x if y != -100) for x in ds["labels"])
        print(f"[mask] {tag}: trainable tokens {keep}/{tot} ({100 * keep / max(1, tot):.1f}%)")

    _stats(trainer.train_dataset, "before tool-output masking")
    trainer.train_dataset = trainer.train_dataset.map(_mask_tool_outputs)
    _stats(trainer.train_dataset, "after tool-output masking")
    ex0 = trainer.train_dataset[0]
    shown = tok.decode([t for t, l in zip(ex0["input_ids"], ex0["labels"]) if l != -100][:400])
    assert "<|tool_response>" not in shown, "tool response still trainable"
    print("[mask] first trainable text:", repr(shown[:300]))
    stats = trainer.train()
    print(stats)

    adapter_dir = Path(args.out) / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    cfg_p = adapter_dir / "adapter_config.json"
    cfg = json.loads(cfg_p.read_text())
    cfg["base_model_name_or_path"] = args.base_model_name
    # vLLM's LoRA loader wants a plain module-name list, not Unsloth's regex string
    if isinstance(cfg.get("target_modules"), str):
        cfg["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cfg_p.write_text(json.dumps(cfg, indent=2))
    print(f"adapter saved → {adapter_dir} (r={cfg.get('r')}, targets={cfg.get('target_modules')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
