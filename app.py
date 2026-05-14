import os
import time
import base64
import httpx

import chainlit as cl
from starlette.responses import JSONResponse

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "google/gemma-4-31b-it:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are an expert plant pathologist. When shown an image of a plant leaf or crop:
1. Identify the disease (or say healthy)
2. Rate confidence (high/medium/low)
3. Explain visible symptoms
4. Suggest actionable remedies

If the image is not a plant/leaf, politely say you can only diagnose plant diseases.
Respond in markdown."""

_start_time = time.time()


async def call_openrouter(messages):
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": MODEL, "messages": messages},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


@cl.on_chat_start
async def start():
    await cl.Message(
        content="🌿 **Plant Disease Diagnosis**\n\nUpload a photo of a plant leaf and I'll identify any disease and suggest remedies.\n\nYou can also ask follow-up questions about the diagnosis."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    images = [f for f in (message.elements or []) if f.mime and f.mime.startswith("image/")]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if images:
        img_data = images[0].content if images[0].content else open(images[0].path, "rb").read()
        b64 = base64.b64encode(img_data).decode()
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{images[0].mime};base64,{b64}"}},
                {"type": "text", "text": message.content or "Diagnose this plant leaf."},
            ],
        })
    elif message.content:
        messages.append({"role": "user", "content": message.content})
    else:
        await cl.Message(content="Please upload an image or ask a question.").send()
        return

    try:
        result = await call_openrouter(messages)
        await cl.Message(content=result).send()
    except httpx.HTTPStatusError as e:
        await cl.Message(content=f"⚠️ API error: {e.response.status_code} — {e.response.text}").send()


# /health endpoint
from chainlit.server import app

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "uptime_seconds": int(time.time() - _start_time)})
