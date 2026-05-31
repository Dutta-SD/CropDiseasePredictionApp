---
title: Crop Disease Diagnosis
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Crop Disease Diagnosis

Chainlit app that diagnoses plant leaf diseases from a photo and suggests treatment.

Upload a photo of a leaf in the chat — the app responds with disease, confidence, severity, and a treatment plan.

## Local development

```bash
cp .env.example .env
# fill in OPENROUTER_API_KEY (free key at https://openrouter.ai/keys)

pip install -r requirements.txt
chainlit run app.py
```

App listens on `http://localhost:8000` by default.

## Lint and format

Ruff handles both linting and import sorting (it replaces the standalone
`isort` tool — same author, single tool, no config conflicts).

```bash
pip install -r requirements-dev.txt
ruff check .              # lint (E/W/F/I/B/UP/SIM/RUF rules)
ruff check --fix .        # apply auto-fixes
ruff format .             # format (Black-compatible style)
ruff format --check .     # verify formatted, no changes
```

Configuration lives in `pyproject.toml`.

## Run with Docker

```bash
docker build -t crop-disease .
docker run --rm -p 7860:7860 -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY crop-disease
```

App listens on `http://localhost:7860`.

## Deploy to Hugging Face Spaces

GitHub is the canonical repo. The HF Space is a deploy target — push to both with one extra command.

One-time setup (already done for `sdutta28/crop-disease-diagnosis`):

1. Create the Space at <https://huggingface.co/new-space> with **Docker** SDK.
2. Add `OPENROUTER_API_KEY` under **Settings → Variables and secrets**.
3. Add the Space as a second git remote:

   ```bash
   git remote add hf https://huggingface.co/spaces/<user>/<space-name>
   ```

Each deploy:

```bash
git push origin mainline             # canonical → GitHub
git push hf mainline:main            # deploy    → HF Space (rebuilds Docker image)
```

The Space's branch is `main` while local branch is `mainline`; the `mainline:main` mapping handles that.

Live at: <https://sdutta28-crop-disease-diagnosis.hf.space>

## Stack

- [Chainlit](https://chainlit.io) — chat UI
- [OpenRouter](https://openrouter.ai) — LLM API (free vision-capable model)
- Docker for HF Spaces deployment
