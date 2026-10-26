## StudioRK3dps-Community-App-API-Workflow-Adaptation

Welcome to StudioRK3dps Community App & API Workflow Adaptation Streamlit app for turning text prompts into images and then converting those images into 3D models.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://studiork3dps-community-app-7ujeaa3i9kknheoguwqw8d.streamlit.app/)

## Features

- Image generation with OpenAI or SDXL via Hugging Face
- 3D model generation with Stability AI or Tripo3D
- Interactive GLB viewer in the browser
- AI chat assistant powered by Cohere for idea generation and workflow guidance
- Cohere-powered 3D image prompt ideas
- PWA metadata and service worker support
- App allows Api keys to be entered on frontend side bar on the left after clicking left arrow for session based usage

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

**Important:** Never commit API keys to version control. Set environment variables in a local `.env` file (ensure `.env` is in `.gitignore`) Or Streamlit Cloud secrets. **Do not commit the `.env` file.**

Example `.env` file:

```bash
OPENAI_API_KEY=your_openai_key
STABILITY_KEY=your_stability_key
HF_TOKEN=your_huggingface_token
TRIPO3D_API_KEY=your_tripo3d_key
COHERE_API_KEY=your_cohere_key
```

Start the app:

```bash
streamlit run streamlit_app.py
```

## Troubleshooting Image Generation and 3D Generation

- If you see an OpenAI error like `billing_hard_limit_reached`, the active key/project has no remaining billing capacity.
- Rotate to a personal paid funded key and update `OPENAI_API_KEY` in your local `.env` (or Streamlit secrets or frontend).
- The app re-reads `.env` during key lookup, so key rotation is picked up on the next request.
- As a fall-back, switch image mode to `Slow_Mode_SDXL` corresponds with huggingface Tokens.
- Triposr allows for free trial usage for Api key
- -Stability Api key is personal paid usage 

## Streamlit Cloud

Use `streamlit_app.py` as the app entry point and add API keys in Streamlit secrets or environment variables. The repo is structured so it can be connected directly to Streamlit Cloud.

## Attributions to API tech providers and frameworks!
