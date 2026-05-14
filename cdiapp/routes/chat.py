import chainlit as cl

from cdiapp.prompts import SYSTEM_PROMPT
from cdiapp.components.llm import call_llm, RateLimitError
from cdiapp.utils.image import encode_image


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
        messages.append({
            "role": "user",
            "content": [
                encode_image(img_data, images[0].mime),
                {"type": "text", "text": message.content or "Diagnose this plant leaf."},
            ],
        })
    elif message.content:
        messages.append({"role": "user", "content": message.content})
    else:
        await cl.Message(content="Please upload an image or ask a question.").send()
        return

    try:
        result = await call_llm(messages)
        await cl.Message(content=result).send()
    except RateLimitError:
        await cl.Message(content="⏳ The AI service is busy right now. Please try again in a minute.").send()
    except Exception:
        await cl.Message(content="⚠️ Something went wrong. Please try again.").send()
