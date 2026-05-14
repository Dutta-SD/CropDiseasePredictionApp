"""CropDiseaseApp — Chainlit entrypoint."""

from cdiapp.routes.chat import *  # noqa: F401,F403 — registers Chainlit handlers
from cdiapp.routes.health import *  # noqa: F401,F403 — registers /health endpoint
