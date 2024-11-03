from enum import StrEnum, auto
from typing import Dict

from fastapi import HTTPException, UploadFile
from pydantic import BaseModel


class ResponseStatus(StrEnum):
    SUCCESS = auto()
    ERROR = auto()


class APIBaseResponse(BaseModel):
    status: ResponseStatus = ResponseStatus.SUCCESS
    message: Dict[str, str]


def validate_image_received(image: UploadFile):
    content_type = image.content_type
    if content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="File received is not an image")
