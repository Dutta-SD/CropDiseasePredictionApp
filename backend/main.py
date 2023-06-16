import logging
import os
from code.apis.model import prediction_router
from code.apis.validation import validation_router
from code.db.connection import db_connection
import uvicorn
from fastapi import FastAPI
from code.config import app_config
from code.utilities.log_utils import get_logger


def add_routes(app: FastAPI) -> FastAPI:
    """Adds API routers to the main APP

    Args:
        app (FastAPI): Main App Component

    Returns:
        FastAPI: App Component with Routers attached
    """
    app.include_router(prediction_router)
    app.include_router(validation_router)
    return app


def get_app() -> FastAPI:
    """Prepares main application

    Returns:
        FastAPI: Main Application
    """
    main_application = FastAPI(
        title=app_config.APP_NAME, version=app_config.APP_VERSION
    )
    main_application = add_routes(main_application)
    return main_application


main_application = get_app()
LOG = get_logger()


@main_application.on_event("startup")
def setup() -> None:
    """Initialization"""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    db_connection.ping()
    LOG.info("SETUP completed")


if __name__ == "__main__":
    uvicorn.run(
        "main:main_application", host="0.0.0.0", port=app_config.PORT, reload=True
    )
