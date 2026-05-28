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

## Run with Docker

```bash
docker build -t crop-disease .
docker run --rm -p 7860:7860 -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY crop-disease
```

App listens on `http://localhost:7860`.

## Deploy to Hugging Face Spaces

1. Create a new Space with **Docker** SDK at https://huggingface.co/new-space.
2. In the Space's **Settings → Variables and secrets**, add `OPENROUTER_API_KEY`.
3. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/<space-name>
   git push space main
   ```
4. The Space auto-builds the Dockerfile and exposes the app on port `7860`.

## Stack

- [Chainlit](https://chainlit.io) — chat UI
- [OpenRouter](https://openrouter.ai) — LLM API (free vision-capable model)
- Docker for HF Spaces deployment
