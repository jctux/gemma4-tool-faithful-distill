"""Build the 9 harness tool schemas exactly as the student sees them.

ADK derives each tool's FunctionDeclaration from the swegemma tool closure's signature and
docstring; we reuse ADK's own parser so the training prompt matches the eval prompt.

  python -m distill.tools_schema > distill/tools_schema.json
"""

from __future__ import annotations

import json
import sys
from typing import Any

from google.adk.tools.function_tool import FunctionTool
from swegemma.context import SwegemmaContext
from swegemma.tools import create_tools

TOOL_ORDER = (
    "run_command",
    "read_file",
    "edit_file",
    "write_file",
    "get_status",
    "submit_patch",
    "get_code_neighbors",
    "search_similar_code",
    "get_code_subgraph",
)


def _schema_to_json(schema: Any) -> dict[str, Any]:
    """google.genai Schema → plain JSON schema dict (OpenAI parameters style)."""
    if schema is None:
        return {"type": "object", "properties": {}}
    d = schema.model_dump(exclude_none=True) if hasattr(schema, "model_dump") else dict(schema)

    def conv(node: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        t = node.get("type")
        if t is not None:
            out["type"] = str(t).lower().replace("type.", "")
        for k in ("description", "enum", "format"):
            if k in node:
                out[k] = node[k]
        if "nullable" in node and node["nullable"]:
            out["nullable"] = True
        if "properties" in node:
            out["properties"] = {k: conv(v) for k, v in node["properties"].items()}
        if "items" in node:
            out["items"] = conv(node["items"])
        if "required" in node:
            out["required"] = node["required"]
        if "any_of" in node:
            out["anyOf"] = [conv(v) for v in node["any_of"]]
        return out

    return conv(d)


def build_schemas(names: tuple[str, ...] = TOOL_ORDER) -> list[dict[str, Any]]:
    ctx = SwegemmaContext()  # no sandbox needed just to introspect
    tools = create_tools(ctx)
    out = []
    for name in names:
        fn = tools[name]
        decl = FunctionTool(fn)._get_declaration()
        out.append(
            {
                "type": "function",
                "function": {
                    "name": decl.name,
                    "description": decl.description or "",
                    "parameters": _schema_to_json(decl.parameters),
                },
            }
        )
    return out


if __name__ == "__main__":
    json.dump(build_schemas(), sys.stdout, indent=2)
    print()
