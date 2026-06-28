# Image to 3D Generator

Streamlit app for turning text prompts into images and then converting those images into 3D models.

## Features

- Image generation with OpenAI or SDXL via Hugging Face
- 3D model generation with Stability AI or Tripo3D
- Interactive GLB viewer in the browser
- AI chat assistant powered by Cohere for idea generation and workflow guidance
- Cohere-powered blog writer tab for generating Markdown blog drafts from a question/topic
- Cohere-powered prompt examples tab for generating reusable prompt ideas from a topic
- PWA metadata and service worker support

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

**Important:** Never commit API keys to version control. Set environment variables in a local `.env` file (ensure `.env` is in `.gitignore`). **Do not commit the `.env` file.**

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

## Streamlit Cloud

Use `streamlit_app.py` as the app entry point and add API keys in Streamlit secrets or environment variables. The repo is structured so it can be connected directly to Streamlit Cloud.