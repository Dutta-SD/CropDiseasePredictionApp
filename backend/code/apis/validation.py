from fastapi import UploadFile
from fastapi.routing import APIRouter
from typing import Dict
from code.config import app_config
import code.service.db as db_functions
import code.utilities.api_utils as api_utils


validation_router = APIRouter(prefix=f"/{app_config.API_V1}/validation")


@validation_router.post("", status_code=200, tags=["validation"])
async def get_validation_token(img: UploadFile) -> Dict:
    # Just validate the token. No need for
    # For now just return a token. Add logic to validate
    #  based on many parameters.
    key, value = db_functions.generate_key_token()
    key = db_functions.store_in_db(key, value)
    return api_utils.success_response(key)
