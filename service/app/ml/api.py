from fastapi import APIRouter, File, UploadFile

from app.definition import APIBaseResponse, ResponseStatus, validate_image_received

router = APIRouter(prefix="/ml", tags=["ML"])


@router.post("/classify")
async def process_image(image: UploadFile = File(...)):
    validate_image_received(image)
    image_bytes = await image.read()
    return APIBaseResponse(
        status=ResponseStatus.SUCCESS,
        message={"predictedClass": "Late Blight"},
    )


@router.post("/can-classify")
async def process_images(image: UploadFile = File(...)):
    validate_image_received(image)
    image_bytes = await image.read()
    return APIBaseResponse(
        status=ResponseStatus.SUCCESS,
        message={"inDistribution": "True"},
    )
