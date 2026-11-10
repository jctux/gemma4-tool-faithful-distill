# Gemma 4 chat/tool format (from google/gemma-4-E4B-it-qat-w4a16-ct chat_template.jinja, 2026-10-25)

- Turns: `<|turn>system\n…<turn|>\n`, `<|turn>user\n…<turn|>\n`, `<|turn>model\n…<turn|>\n`.
- Tool declarations go in the system turn as `declaration:<name>{description:<|"|>…<|"|>,parameters:{…}}` (template macro `format_function_declaration`), generated automatically when `tools=[…]` (OpenAI function schema) is passed to `apply_chat_template`.
- Model tool call: `<|tool_call>call:<name>{key:value,…}<tool_call|>` — keys `dictsort`ed, values JSON-ish with `<|"|>` string quoting.
- Tool result: `<|tool_response>…<tool_response|>` (template macro `format_tool_response_block`), from `role: tool` messages matched by `tool_call_id`, or legacy `tool_responses` on the assistant message.
- Thinking: `<|channel>thought\n…<channel|>` before the visible answer; only emitted for turns after the last user message (`thinking_gate`) unless `preserve_thinking`. vLLM `reasoning_parser=gemma4` strips it; harness passes `include_thoughts` to keep it in context.
- EOS ids: 1, 106 (`<turn|>`), 50.

## Implication for distill/to_sft.py
- Store trajectories as OpenAI-style `messages` (system / user / assistant{content, tool_calls[{id, function{name, arguments}}]} / tool{tool_call_id, content}) + `tools` schema list (the 9 harness tools, same descriptions as the student's ADK registry).
- Render with `tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)` at train time; mask loss to model turns only (`<|turn>model\n` … `<turn|>`).
- Decide thinking policy: (a) no thinking (`thinking_budget: 0` at eval) → drop teacher reasoning, or (b) short thought = teacher's visible text before each tool call, wrapped as `<|channel>thought…<channel|>`. Start with (b) capped at ~300 tokens; ablate.
- Tool outputs are already truncated by the harness (5000 chars / 150 lines) so the student sees the same distribution.

## Model geometry (E4B ct)
42 layers, hidden 2560, 8 heads / 2 kv heads, 18 kv-shared layers, sliding window 512, vocab 262144, PLE 256-d per layer (bf16, NOT quantized: ~5.6 GB), W4 group-32 symmetric on Linear (vision/audio/lm_head excluded). ⇒ ~10 GB resident on GPU before KV cache; 12 GB 3060 is tight → try `MAX_LEN=16384`, else E2B.

## Validated 2026-10-25 16:45 (transformers + model chat_template.jinja)
- `distill/to_sft.py` rebuilds API-level turns from Claude Code's pipelined stream (text/thinking/tool_use events merged; parallel tool calls + interleaved results paired by id; harness JSON preferred for results).
- `tool_calls[].function.arguments` must be a **dict** for the template. Rendering works: `<|tool_call>call:run_command{command:<|"|>…<|"|>}<tool_call|><|tool_response>response:run_command{value:<|"|>{…}<|"|>}<tool_response|>`; declarations use uppercase types (STRING).
- Teacher text before a call is dropped by default; pass `preserve_thinking=True` to render it as `<|channel>thought…<channel|>` (train default; `--no-thinking` ablation).
- Lengths on first 14 episodes: median ~11k tokens, 5/14 > 16k, max 26.7k ⇒ train with max_seq_len 24576 on L4 (24 GB) QLoRA.
