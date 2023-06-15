from fastapi.routing import APIRouter
from fastapi.requests import Request
from typing import Dict


prediction_router = APIRouter(prefix="/model")


@prediction_router.get("", status_code=200)
async def get_model_homepage() -> Dict:
    """Fetch Sample Output to check if API working or not

    Returns:
        Dict: JSON response
    """
    return {"response": "Working GET @ /model"}
