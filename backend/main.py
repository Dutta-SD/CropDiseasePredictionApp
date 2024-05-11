import uvicorn
from fastapi import FastAPI, Request

from app.definition import APIBaseResponse, ResponseStatus
from app.ml.api import router

server = FastAPI()

server.include_router(router)


@server.exception_handler(Exception)
async def handle_exception(request: Request, exc: Exception):
    return APIBaseResponse(status=ResponseStatus.ERROR, message=str(exc))


if __name__ == "__main__":
    uvicorn.run("main:server", host="0.0.0.0", port=8000, reload=True)
