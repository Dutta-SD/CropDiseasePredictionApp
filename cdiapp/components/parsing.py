"""Tolerant JSON extraction from LLM responses.

Even with explicit "no markdown fences" instructions, models routinely wrap
JSON in ```json ... ``` or add a trailing sentence. We extract the first
balanced top-level object and let pydantic decide if it's valid.
"""

from __future__ import annotations

import json
import re

from pydantic import TypeAdapter

from cdiapp.schema import DiagnosisOutput, SchemaViolationError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

_adapter: TypeAdapter[DiagnosisOutput] = TypeAdapter(DiagnosisOutput)


def _extract_json_object(text: str) -> str:
    """Return the first balanced {...} block in `text`, or `text` itself."""
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1)

    start = text.find("{")
    if start == -1:
        return text.strip()

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:].strip()


def parse_diagnosis(raw: str) -> DiagnosisOutput:
    """Parse and validate a raw LLM response. Raises SchemaViolationError on failure."""
    candidate = _extract_json_object(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise SchemaViolationError(raw, exc) from exc

    try:
        return _adapter.validate_python(data)
    except Exception as exc:
        raise SchemaViolationError(raw, exc) from exc
