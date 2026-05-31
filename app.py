"""CropDiseaseApp — Chainlit entrypoint.

Chainlit registers its decorated handlers (`@cl.on_chat_start`, `@cl.on_message`,
`@cl.set_starters`, `@app.get(...)`) as a side effect of being imported, so we
star-import the route modules here. Ruff is configured to allow F401/F403 in
this file via `per-file-ignores`.
"""

from cdiapp.routes.chat import *
from cdiapp.routes.health import *
