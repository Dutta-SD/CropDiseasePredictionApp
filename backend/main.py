import os
from code.apis.prediction import prediction_router
import uvicorn
from fastapi import FastAPI
from code.config import Settings


def setup() -> None:
    """Initialization"""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def add_routes(app: FastAPI) -> FastAPI:
    """Adds API routers to the main APP

    Args:
        app (FastAPI): Main App Component

    Returns:
        FastAPI: App Component with Routers attached
    """
    app.include_router(prediction_router)
    return app


def get_app() -> FastAPI:
    """Prepares main application

    Returns:
        FastAPI: Main Application
    """
    main_application = FastAPI(title=Settings.APP_NAME, version=Settings.APP_VERSION)
    main_application = add_routes(main_application)
    return main_application


main_application = get_app()


if __name__ == "__main__":
    setup()
    uvicorn.run(
        "main:main_application",
        host="0.0.0.0",
        port=Settings.PORT,
        reload=True,
    )
