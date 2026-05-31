import logging

import chainlit as cl

from cdiapp.components.llm import RateLimitError, call_llm, call_llm_typed
from cdiapp.prompts import FOLLOWUP_PROMPT, SYSTEM_PROMPT
from cdiapp.render import render
from cdiapp.schema import SchemaViolationError
from cdiapp.utils.image import encode_image

log = logging.getLogger(__name__)

WELCOME = (
    "🌿 **Plant Disease Triage**\n\n"
    "Upload a photo of a plant leaf and I'll identify likely problems and suggest "
    "next steps. I don't prescribe treatments — for chemicals, doses, or anything "
    "you're unsure about, please consult your local extension officer (KVK) or "
    "qualified agronomist.\n\n"
    "You can ask follow-up questions about the diagnosis."
)


@cl.on_chat_start
async def start():
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message):
    images = [f for f in (message.elements or []) if f.mime and f.mime.startswith("image/")]

    if images:
        await _handle_image(message, images[0])
    elif message.content:
        await _handle_text_followup(message.content)
    else:
        await cl.Message(content="Please upload an image or ask a question.").send()


async def _handle_image(message: cl.Message, image) -> None:
    img_data = image.content if image.content else open(image.path, "rb").read()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                encode_image(img_data, image.mime),
                {"type": "text", "text": message.content or "Diagnose this plant leaf."},
            ],
        },
    ]

    try:
        diagnosis = await call_llm_typed(messages)
    except RateLimitError:
        await cl.Message(content="⏳ The AI service is busy right now. Please try again in a minute.").send()
        return
    except SchemaViolationError:
        await cl.Message(
            content=(
                "⚠️ I couldn't produce a structured diagnosis for this image. "
                "Try a clearer, well-lit photo of a single leaf, and ask again."
            )
        ).send()
        return
    except Exception:
        log.exception("on_message.image_failure")
        await cl.Message(content="⚠️ Something went wrong. Please try again.").send()
        return

    await cl.Message(content=render(diagnosis)).send()


async def _handle_text_followup(text: str) -> None:
    """Free-form follow-up turn (no image). Not schema-validated.

    The triage rules in the system prompt still apply — the model is told not
    to prescribe doses or chemicals.
    """
    messages = [
        {"role": "system", "content": FOLLOWUP_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        reply = await call_llm(messages)
    except RateLimitError:
        await cl.Message(content="⏳ The AI service is busy right now. Please try again in a minute.").send()
        return
    except Exception:
        log.exception("on_message.text_failure")
        await cl.Message(content="⚠️ Something went wrong. Please try again.").send()
        return

    await cl.Message(content=reply).send()
