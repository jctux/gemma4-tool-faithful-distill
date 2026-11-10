"""Build a submission dir for local eval: python eval/make_sub.py NAME [--adapter DIR --adapter-name N] [--thinking 1024]"""
import argparse, shutil, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--adapter", default=None); ap.add_argument("--adapter-name", default=None)
ap.add_argument("--thinking", type=int, default=1024); ap.add_argument("--freq-penalty", type=float, default=0.0); ap.add_argument("--temperature", type=float, default=0.3); ap.add_argument("--model", default="gemma-4-e2b-it-qat-w4a16-ct")
ap.add_argument("--out-root", default=str(ROOT / "eval_subs"))
a = ap.parse_args()
d = Path(a.out_root) / a.name
shutil.rmtree(d, ignore_errors=True); (d / "prompts").mkdir(parents=True); (d / "configs").mkdir()
shutil.copy(ROOT / "submission/prompts/system.md", d / "prompts/system.md")
adapter_line = f"adapter: {a.adapter_name}\n" if a.adapter else ""
(d / "agent.yaml").write_text(f"""name: swe_agent
model: {a.model}
{adapter_line}instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
generate_content_config: !include configs/sampling.yaml
""")
fp = f"frequency_penalty: {a.freq_penalty}\n" if a.freq_penalty else ""
think = f"thinking_config:\n  thinking_budget: {a.thinking}\n  include_thoughts: true\n" if a.thinking > 0 else "thinking_config:\n  thinking_budget: 0\n"
(d / "configs/sampling.yaml").write_text(f"temperature: {a.temperature}\ntop_p: 0.95\nmax_output_tokens: 4096\n{fp}{think}")
(d / "eval_config.yaml").write_text("evaluation:\n  timeout_seconds: 120\n  max_tool_calls: 40\n  max_time_minutes: 8\n  max_turns: 60\n")
if a.adapter:
    (d / "adapters" / a.adapter_name).mkdir(parents=True)
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(Path(a.adapter) / f, d / "adapters" / a.adapter_name / f)
print("built", d)
