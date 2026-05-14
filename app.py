import os
import time

import chainlit as cl
from google import genai
from starlette.responses import JSONResponse

# Configure Gemini client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = """You are an expert plant pathologist. When shown an image of a plant leaf or crop:
1. Identify the disease (or say healthy)
2. Rate confidence (high/medium/low)
3. Explain visible symptoms
4. Suggest actionable remedies

If the image is not a plant/leaf, politely say you can only diagnose plant diseases.
Respond in markdown."""

_start_time = time.time()


@cl.on_chat_start
async def start():
    await cl.Message(
        content="🌿 **Plant Disease Diagnosis**\n\nUpload a photo of a plant leaf and I'll identify any disease and suggest remedies.\n\nYou can also ask follow-up questions about the diagnosis."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    images = [f for f in (message.elements or []) if f.mime and f.mime.startswith("image/")]

    parts = [SYSTEM_PROMPT]

    if images:
        img_data = images[0].content if images[0].content else open(images[0].path, "rb").read()
        parts.append(genai.types.Part.from_bytes(data=img_data, mime_type=images[0].mime))
        parts.append(message.content or "Diagnose this plant leaf.")
    elif message.content:
        parts.append(message.content)
    else:
        await cl.Message(content="Please upload an image or ask a question.").send()
        return

    response = client.models.generate_content(model=MODEL, contents=parts)
    await cl.Message(content=response.text).send()


# /health endpoint
from chainlit.server import app

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "uptime_seconds": int(time.time() - _start_time)})
