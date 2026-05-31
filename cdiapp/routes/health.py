import time

from chainlit.server import app
from starlette.responses import JSONResponse

_start_time = time.time()


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "uptime_seconds": int(time.time() - _start_time)})
