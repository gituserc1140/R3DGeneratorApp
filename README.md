## StudioRK3dps-Community-App-API-Workflow-Adaptation

Welcome to StudioRK3dps Community App & API Workflow Adaptation Streamlit app for turning text prompts into images and then converting those images into 3D models.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://studiork3dps-community-app-7ujeaa3i9kknheoguwqw8d.streamlit.app/)

Attributions to API tech providers and frameworks!

## Check out my socials!

[![TikTok](https://img.shields.io/badge/TikTok-@studiork3dps-000000?style=for-the-badge&logo=tiktok&logoColor=white)](https://www.tiktok.com/@studiork3dps)
[![Instagram](https://img.shields.io/badge/Instagram-@rk3dps-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/rk3dps/)

## Features

- Image generation with OpenAI
- 3D model generation with Stability AI
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
OPENAI_IMAGE_MODEL=gpt-image-1
STABILITY_KEY=your_stability_key
COHERE_API_KEY=your_cohere_key
```

Start the app:

```bash
streamlit run streamlit_app.py
```

## Troubleshooting Image Generation and 3D Generation

- If you see an OpenAI error like `billing_hard_limit_reached`, the active key/project has no remaining billing capacity.
- If image generation reports that the configured model is unavailable, set `OPENAI_IMAGE_MODEL` to a model supported by your OpenAI project (for example `gpt-image-1`, `dall-e-3`, or `dall-e-2`) in your `.env` or Streamlit secrets.
- Rotate to a personal paid funded key and update `OPENAI_API_KEY` in your local `.env` (or Streamlit secrets or frontend).
- The app re-reads `.env` during key lookup, so key rotation is picked up on the next request.
- Stability Api key is personal paid usage 

## Streamlit Cloud

Use `streamlit_app.py` as the app entry point and add API keys in Streamlit secrets or environment variables. The repo is structured so it can be connected directly to Streamlit Cloud.


