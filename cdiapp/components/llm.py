import os
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"


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
            json={"model": MODEL, "messages": messages},
        )
        if resp.status_code == 429:
            raise RateLimitError()
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
