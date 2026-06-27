import logging
import os

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cdiapp.components.parsing import parse_diagnosis
from cdiapp.schema import DiagnosisOutput, SchemaViolationError

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free-tier vision models. The `models` array triggers OpenRouter's fallback
# routing: it tries each in order when a model is rate-limited or unavailable.
# All entries use the :free suffix — no paid models are ever selected.
FREE_VISION_MODELS = [
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "qwen/qwen2.5-vl-7b-instruct:free",
]

log = logging.getLogger(__name__)


class RateLimitError(Exception):
    pass


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=2, max=16),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def call_llm(messages: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"models": FREE_VISION_MODELS, "messages": messages},
        )
        if resp.status_code == 429:
            raise RateLimitError()
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


_CORRECTIVE_INSTRUCTION = (
    "Your previous response did not parse as the required JSON schema. "
    "Error: {error}\n"
    "Return ONLY a single valid JSON object matching the schema in the system prompt. "
    "No markdown fences. No prose. No comments."
)


async def call_llm_typed(messages: list[dict]) -> DiagnosisOutput:
    """Call the LLM and validate the response against DiagnosisOutput.

    On a schema violation, retries ONCE with a corrective instruction that
    includes the validation error. On a second violation, raises
    SchemaViolationError carrying the latest raw response.
    """
    raw = await call_llm(messages)
    try:
        result = parse_diagnosis(raw)
        log.info("llm.schema_ok kind=%s", result.kind)
        return result
    except SchemaViolationError as first_err:
        log.warning("llm.schema_violation attempt=1 error=%s", first_err.validation_error)
        corrective_messages = [
            *messages,
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": _CORRECTIVE_INSTRUCTION.format(error=first_err.validation_error),
            },
        ]
        raw2 = await call_llm(corrective_messages)
        try:
            result = parse_diagnosis(raw2)
            log.info("llm.schema_ok_after_retry kind=%s", result.kind)
            return result
        except SchemaViolationError as second_err:
            log.error("llm.schema_violation attempt=2 error=%s", second_err.validation_error)
            raise
