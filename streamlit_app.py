import base64
import os
import tempfile
import time
import traceback
from collections.abc import Mapping
from pathlib import Path

import requests
import streamlit as st

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return False


load_dotenv()


def _lookup_key_recursive(container, key_name):
    """Recursively find a key in nested mappings."""
    if not isinstance(container, Mapping):
        return None

    lowered = key_name.lower()
    if key_name in container and container[key_name]:
        return container[key_name]
    if lowered in container and container[lowered]:
        return container[lowered]

    for value in container.values():
        if isinstance(value, Mapping):
            found = _lookup_key_recursive(value, key_name)
            if found:
                return found
    return None


def get_api_key(key_name):
    """Get API key from environment or Streamlit secrets."""
    runtime_keys = st.session_state.get("runtime_api_keys", {})
    if key_name in runtime_keys and runtime_keys[key_name]:
        return runtime_keys[key_name]

    lowered = key_name.lower()
    if lowered in runtime_keys and runtime_keys[lowered]:
        return runtime_keys[lowered]

    value = os.environ.get(key_name)
    if value:
        return value

    # Common fallback for lowercase env var naming.
    value = os.environ.get(lowered)
    if value:
        return value

    try:
        secrets_dict = st.secrets.to_dict() if hasattr(st.secrets, "to_dict") else dict(st.secrets)
        return _lookup_key_recursive(secrets_dict, key_name)
    except Exception:
        return None


def get_first_available_key(*key_names):
    """Return the first non-empty key value from the provided key names."""
    for key_name in key_names:
        value = get_api_key(key_name)
        if value:
            return value
    return None


def has_any_key(*key_names):
    """Return True if any key name resolves from env/secrets."""
    return get_first_available_key(*key_names) is not None


def display_3d_model(glb_path):
    """Render a simple success panel for generated GLB files."""
    glb_file = Path(glb_path)
    st.success("3D model generated")
    st.write(f"Saved model file: {glb_file.name}")


st.set_page_config(
    page_title="Image → 3D Generator",
    layout="wide",
    page_icon="🧊",
    initial_sidebar_state="collapsed",
)


def get_openai_client():
    from openai import OpenAI

    api_key = get_first_available_key("OPENAI_API_KEY", "OPENAI_KEY")
    if not api_key:
        raise ValueError("OpenAI key not found. Add OPENAI_API_KEY to Streamlit secrets.")
    return OpenAI(api_key=api_key)


def generate_image_openai(prompt: str) -> str:
    try:
        if not get_first_available_key("OPENAI_API_KEY", "OPENAI_KEY"):
            raise ValueError("OpenAI key not found. Add OPENAI_API_KEY to Streamlit secrets.")

        client = get_openai_client()
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )

        image_item = response.data[0]
        image_base64 = getattr(image_item, "b64_json", None)
        image_url = getattr(image_item, "url", None)

        if image_base64:
            image_bytes = base64.b64decode(image_base64)
        elif image_url:
            image_bytes = requests.get(image_url, timeout=60).content
        else:
            raise Exception("OpenAI response did not include image data")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(image_bytes)
        tmp.close()

        return tmp.name
    except Exception as e:
        st.error(f"Image generation failed: {e}")
        traceback.print_exc()
        return ""


def generate_image_sdxl(prompt: str) -> str:
    try:
        hf_token = get_first_available_key("HF_TOKEN", "HUGGINGFACE_API_KEY", "RKStudioHF1")
        if not hf_token:
            raise ValueError("Set HF_TOKEN (or HUGGINGFACE_API_KEY / RKStudioHF1) in Streamlit secrets.")

        api_url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Accept": "image/png",
        }
        payload = {"inputs": prompt}

        response = requests.post(api_url, headers=headers, json=payload, timeout=180)
        if response.status_code != 200:
            raise Exception(f"Hugging Face SDXL request failed ({response.status_code}): {response.text}")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(response.content)
        tmp.close()
        return tmp.name
    except Exception as e:
        st.error(f"SDXL image generation failed: {e}")
        traceback.print_exc()
        return ""


def image_to_3d(host, image_path, **kwargs):
    stability_key = get_api_key("STABILITY_KEY") or get_api_key("STABILITY_API_KEY")
    if not stability_key:
        raise ValueError("STABILITY_KEY/STABILITY_API_KEY not found in environment or secrets")

    with open(image_path, "rb") as image_file:
        response = requests.post(
            host,
            headers={
                "Authorization": f"Bearer {stability_key}",
                "Accept": "model/gltf-binary",
            },
            files={"image": image_file},
            data=kwargs,
            timeout=180,
        )

    if not response.ok:
        detail = response.text
        if response.status_code == 401:
            raise Exception(f"Stability authentication failed (401). Check STABILITY key. Details: {detail}")
        if response.status_code == 403:
            raise Exception(
                "Stability returned 403 (forbidden). Your API key likely does not have access to the Stable Fast 3D endpoint. "
                f"Details: {detail}"
            )
        raise Exception(f"Stability request failed ({response.status_code}): {detail}")

    return response.content


def generate_3d_model_stability(image_path, texture_resolution, foreground_ratio, remesh, vertex_count):
    try:
        host = "https://api.stability.ai/v2beta/3d/stable-fast-3d"
        glb_data = image_to_3d(
            host=host,
            image_path=image_path,
            texture_resolution=texture_resolution,
            foreground_ratio=foreground_ratio,
            remesh=remesh,
            vertex_count=vertex_count,
        )

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
        tmp.write(glb_data)
        tmp.close()

        return tmp.name
    except Exception as e:
        st.error(f"3D generation failed: {e}")
        traceback.print_exc()
        return ""


def upload_image_to_tripo3d(image_path: str, api_key: str) -> str:
    upload_url = "https://api.tripo3d.ai/v2/openapi/upload/sts"
    headers = {"Authorization": f"Bearer {api_key}"}

    file_extension = os.path.splitext(image_path)[1].lstrip('.').lower()
    mime_type = f"image/{file_extension}" if file_extension in ["jpeg", "jpg", "png"] else "application/octet-stream"

    with open(image_path, "rb") as image_file:
        files = {"file": (os.path.basename(image_path), image_file, mime_type)}
        response = requests.post(upload_url, headers=headers, files=files, timeout=60)

    response.raise_for_status()
    response_json = response.json()
    file_token = response_json.get("data", {}).get("image_token")
    if not file_token:
        raise Exception(f"Could not find image_token in upload response: {response_json}")
    return file_token


def create_tripo3d_task(file_token: str, image_path: str, api_key: str) -> str:
    generation_url = "https://api.tripo3d.ai/v2/openapi/task"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    file_extension = os.path.splitext(image_path)[1].lstrip('.').lower()
    data = {
        "type": "image_to_model",
        "file": {
            "type": file_extension,
            "file_token": file_token,
        },
    }

    response = requests.post(generation_url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    response_json = response.json()
    task_id = response_json.get("data", {}).get("task_id")
    if not task_id:
        raise Exception(f"Could not find task_id in task response: {response_json}")
    return task_id


def poll_tripo3d_task(task_id: str, api_key: str, timeout_seconds: int = 300, interval_seconds: int = 5) -> dict:
    status_url = f"https://api.tripo3d.ai/v2/openapi/task/{task_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        response = requests.get(status_url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()

        status = payload.get("data", {}).get("status", "").lower()
        if status in {"succeeded", "success", "completed", "done"}:
            return payload
        if status in {"failed", "error", "cancelled"}:
            raise Exception(f"Tripo3D task failed: {payload}")

        time.sleep(interval_seconds)

    raise TimeoutError("Timed out waiting for Tripo3D task completion")


def generate_3d_model_tripo(image_path):
    try:
        api_key = get_first_available_key("TRIPO3D_API_KEY", "RKStudioTripo")
        if not api_key:
            raise ValueError("Set TRIPO3D_API_KEY (or RKStudioTripo) in Streamlit secrets.")

        file_token = upload_image_to_tripo3d(image_path, api_key)
        task_id = create_tripo3d_task(file_token, image_path, api_key)
        final_payload = poll_tripo3d_task(task_id, api_key)

        download_url = final_payload.get("data", {}).get("output", {}).get("pbr_model")
        if not download_url:
            raise Exception(f"Failed to retrieve download URL from Tripo3D response: {final_payload}")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
        with requests.get(download_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                tmp.write(chunk)
        tmp.close()

        return tmp.name
    except Exception as e:
        st.error(f"Tripo3D model generation failed: {e}")
        traceback.print_exc()
        return ""


def main():
    st.title("RKstudio Image to 3D Model Generator")
    st.caption("Safe mode: minimized workflow for reliable startup.")

    with st.expander("API key overrides (session only)", expanded=False):
        st.caption("Optional: paste keys here to test immediately. These are not saved to repo and reset when session restarts.")
        runtime_keys = st.session_state.get("runtime_api_keys", {})
        runtime_keys["OPENAI_API_KEY"] = st.text_input(
            "OpenAI key",
            value=runtime_keys.get("OPENAI_API_KEY", ""),
            type="password",
            key="override_openai_api_key",
        ).strip()
        runtime_keys["HF_TOKEN"] = st.text_input(
            "Hugging Face key (HF_TOKEN)",
            value=runtime_keys.get("HF_TOKEN", ""),
            type="password",
            key="override_hf_token",
        ).strip()
        runtime_keys["STABILITY_KEY"] = st.text_input(
            "Stability key",
            value=runtime_keys.get("STABILITY_KEY", ""),
            type="password",
            key="override_stability_key",
        ).strip()
        runtime_keys["TRIPO3D_API_KEY"] = st.text_input(
            "Tripo3D key",
            value=runtime_keys.get("TRIPO3D_API_KEY", ""),
            type="password",
            key="override_tripo3d_api_key",
        ).strip()
        st.session_state["runtime_api_keys"] = runtime_keys

    with st.expander("Runtime status", expanded=True):
        st.write("App initialized successfully.")
        st.write(f"Python: {os.environ.get('PYTHON_VERSION', 'unknown')}")

    # Safe diagnostics: only indicates presence/absence of required keys.
    with st.expander("API key diagnostics (safe)", expanded=True):
        openai_ready = has_any_key("OPENAI_API_KEY", "OPENAI_KEY")
        sdxl_ready = has_any_key("HF_TOKEN", "HUGGINGFACE_API_KEY", "RKStudioHF1")
        stability_ready = has_any_key("STABILITY_KEY", "STABILITY_API_KEY")
        tripo_ready = has_any_key("TRIPO3D_API_KEY", "RKStudioTripo")

        st.write(f"OpenAI key detected: {'yes' if openai_ready else 'no'}")
        st.write(f"SDXL key detected: {'yes' if sdxl_ready else 'no'}")
        st.write(f"Stability key detected: {'yes' if stability_ready else 'no'}")
        st.write(f"Tripo key detected: {'yes' if tripo_ready else 'no'}")
        st.caption("Only key presence is shown. Secret values are never displayed.")

    tab1, tab2 = st.tabs(["Image", "3D Model"])

    with tab1:
        st.subheader("Step 1: Create or upload an image")
        mode = st.selectbox(
            "Image generation mode",
            ["Fast_Mode_OA", "Slow_Mode_SDXL"],
            index=0,
            key="image_mode_select",
        )
        prompt = st.text_area(
            "Image prompt",
            value="Simple toy model of a chicken on a plain studio background",
            key="image_prompt_input",
        )

        if st.button("Generate Image", key="generate_image_button"):
            with st.spinner("Generating image..."):
                image_path = generate_image_openai(prompt) if mode == "Fast_Mode_OA" else generate_image_sdxl(prompt)
                if image_path:
                    st.session_state["image_path"] = image_path
                    st.image(image_path, caption="Generated Image", width=400)

        uploaded = st.file_uploader("Or upload image", type=["png", "jpg", "jpeg"], key="upload_image_file")
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(uploaded.read())
            tmp.close()
            st.session_state["image_path"] = tmp.name
            st.image(tmp.name, caption="Uploaded Image", width=400)

    with tab2:
        st.subheader("Step 2: Convert image to 3D")
        image_path = st.session_state.get("image_path")
        if not image_path:
            st.info("Generate or upload an image in Step 1 first.")
        else:
            st.image(image_path, caption="Input Image", width=320)
            mode = st.selectbox(
                "3D mode",
                ["Medium_Quality_Mode_STA", "High_Quality_Mode_TRO"],
                index=0,
                key="model_mode_select",
            )

            if mode == "Medium_Quality_Mode_STA":
                texture_resolution = st.selectbox(
                    "Texture Resolution",
                    ["512", "1024", "2048"],
                    index=1,
                    key="texture_resolution_select",
                )
                foreground_ratio = st.slider(
                    "Foreground Ratio",
                    0.1,
                    1.0,
                    0.85,
                    0.05,
                    key="foreground_ratio_slider",
                )
                remesh = st.selectbox("Remesh", ["none", "quad", "triangle"], key="remesh_select")
                vertex_count = st.number_input("Vertex Count (-1 = auto)", value=-1, key="vertex_count_input")

                if st.button("Generate 3D Model", key="generate_3d_stability_button"):
                    with st.spinner("Generating 3D model..."):
                        glb_path = generate_3d_model_stability(
                            image_path,
                            texture_resolution,
                            foreground_ratio,
                            remesh,
                            vertex_count,
                        )
                        if glb_path:
                            display_3d_model(glb_path)
                            with open(glb_path, "rb") as f:
                                st.download_button(
                                    "Download GLB",
                                    f,
                                    file_name="model.glb",
                                    mime="model/gltf-binary",
                                    key="download_glb_stability",
                                )
            else:
                if st.button("Generate 3D Model", key="generate_3d_tripo_button"):
                    with st.spinner("Generating 3D model..."):
                        glb_path = generate_3d_model_tripo(image_path)
                        if glb_path:
                            display_3d_model(glb_path)
                            with open(glb_path, "rb") as f:
                                st.download_button(
                                    "Download GLB",
                                    f,
                                    file_name="model.glb",
                                    mime="model/gltf-binary",
                                    key="download_glb_tripo",
                                )


def run_app():
    try:
        main()
    except Exception as exc:
        st.error("Startup error detected. Details are shown below.")
        st.exception(exc)
        st.code(traceback.format_exc())


if __name__ == "__main__":
    run_app()