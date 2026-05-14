import base64


def encode_image(content: bytes, mime_type: str) -> dict:
    b64 = base64.b64encode(content).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
