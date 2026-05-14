import time

from starlette.responses import JSONResponse
from chainlit.server import app

_start_time = time.time()


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "uptime_seconds": int(time.time() - _start_time)})
