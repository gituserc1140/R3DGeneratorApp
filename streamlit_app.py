import os
import tempfile
import time
import traceback
import io
import math
import base64

import plotly.graph_objects as go
import requests
import streamlit as st
import trimesh
from dotenv import load_dotenv
from PIL import Image, ImageDraw


load_dotenv()


def get_api_key(key_name):
    """Get API key from environment or Streamlit secrets."""
    value = os.environ.get(key_name)
    if value:
        return value

    try:
        return st.secrets[key_name]
    except Exception:
        return None


def _project_point_iso(point):
    """Project a 3D point into a 2D isometric-like view."""
    x, y, z = point
    cos_30 = math.sqrt(3) / 2
    sin_30 = 0.5
    return ((x - y) * cos_30, (x + y) * sin_30 - z)


def build_static_snapshot(glb_path, note=""):
    """Render a static PNG snapshot from mesh vertices when triangulation fails."""
    width, height = 960, 640
    image = Image.new("RGB", (width, height), "#eef2ff")
    draw = ImageDraw.Draw(image)

    loaded = trimesh.load(glb_path, force="scene")
    meshes = loaded.dump(concatenate=False) if isinstance(loaded, trimesh.Scene) else [loaded]

    points = []
    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        vertices = getattr(mesh, "vertices", None)
        if vertices is None or len(vertices) == 0:
            continue

        sample_step = max(1, len(vertices) // 2500)
        for point in vertices[::sample_step]:
            points.append(point)

    if not points:
        draw.rectangle((28, 28, width - 28, height - 28), outline="#94a3b8", width=2)
        draw.text((44, 44), "Static snapshot unavailable", fill="#0f172a")
        if note:
            draw.text((44, 74), f"Reason: {note[:120]}", fill="#334155")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    projected = [_project_point_iso(point) for point in points]
    xs = [pt[0] for pt in projected]
    ys = [pt[1] for pt in projected]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max(max_x - min_x, 1e-9)
    dy = max(max_y - min_y, 1e-9)

    margin = 56
    sx = (width - (2 * margin)) / dx
    sy = (height - (2 * margin)) / dy
    scale = min(sx, sy)

    draw.rectangle((24, 24, width - 24, height - 24), outline="#94a3b8", width=2)
    for x, y in projected:
        px = margin + (x - min_x) * scale
        py = margin + (y - min_y) * scale
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill="#0f172a")

    draw.text((36, 34), "Static snapshot fallback", fill="#0f172a")
    if note:
        draw.text((36, 58), f"Reason: {note[:120]}", fill="#334155")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_placeholder_snapshot(note=""):
    """Return a simple placeholder PNG when model snapshot generation fails."""
    width, height = 960, 640
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, width - 24, height - 24), outline="#94a3b8", width=2)
    draw.text((40, 44), "Model image preview unavailable", fill="#0f172a")
    if note:
        draw.text((40, 74), f"Reason: {note[:120]}", fill="#334155")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def update_snapshot_preview(glb_path):
    """Store a PNG snapshot preview in session for reliable display."""
    try:
        st.session_state["glb_preview_png"] = build_static_snapshot(glb_path)
        st.session_state["glb_preview_error"] = ""
    except Exception as exc:
        st.session_state["glb_preview_png"] = build_placeholder_snapshot(str(exc))
        st.session_state["glb_preview_error"] = str(exc)


def show_preview_mode_badge(mode):
    """Display a compact badge indicating the active preview mode."""
    palette = {
        "Interactive": ("#166534", "#dcfce7", "#bbf7d0"),
        "Static": ("#7c2d12", "#ffedd5", "#fed7aa"),
    }
    text_color, bg_color, border_color = palette.get(mode, ("#0f172a", "#e2e8f0", "#cbd5e1"))
    st.markdown(
        (
            f"<div style='display:inline-block;padding:0.15rem 0.55rem;border-radius:999px;"
            f"font-size:0.78rem;font-weight:600;color:{text_color};background:{bg_color};"
            f"border:1px solid {border_color};margin-bottom:0.35rem;'>"
            f"Preview: {mode}</div>"
        ),
        unsafe_allow_html=True,
    )


def build_plotly_figure(glb_path):
    """Build a Plotly mesh figure from a GLB file."""
    loaded = trimesh.load(glb_path, force="scene")
    meshes = loaded.dump(concatenate=False) if isinstance(loaded, trimesh.Scene) else [loaded]

    figure = go.Figure()
    has_geometry = False
    failed_meshes = 0

    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces is None or len(mesh.faces) == 0:
            continue

        try:
            triangles = mesh.triangles
            if triangles is None or len(triangles) == 0:
                failed_meshes += 1
                continue
            vertices = triangles.reshape((-1, 3))
            face_count = len(triangles)
            faces = [(idx, idx + 1, idx + 2) for idx in range(0, face_count * 3, 3)]
        except Exception:
            failed_meshes += 1
            continue

        i = [item[0] for item in faces]
        j = [item[1] for item in faces]
        k = [item[2] for item in faces]
        color = "#ff6b6b"
        visual = getattr(mesh, "visual", None)
        material = getattr(visual, "material", None) if visual is not None else None
        base_color = getattr(material, "baseColorFactor", None)
        if base_color and len(base_color) >= 3:
            rgb = tuple(int(channel) for channel in base_color[:3])
            color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=i,
                j=j,
                k=k,
                color=color,
                flatshading=False,
                lighting={"ambient": 0.55, "diffuse": 0.85, "specular": 0.2, "roughness": 0.7},
                lightposition={"x": 120, "y": 160, "z": 200},
                hoverinfo="skip",
                name=os.path.basename(glb_path),
                showscale=False,
            )
        )
        has_geometry = True

    if not has_geometry:
        if failed_meshes:
            raise ValueError("GLB mesh could not be triangulated cleanly")
        raise ValueError("No mesh geometry found in GLB file")

    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#f8fafc",
        scene={
            "bgcolor": "#f8fafc",
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "camera": {"eye": {"x": 1.8, "y": 1.8, "z": 1.2}},
        },
        showlegend=False,
    )
    return figure


def display_3d_model(glb_path, chart_key=None):
    """Display a 3D GLB model using Plotly as a browser-safe fallback."""
    st.caption(f"Viewing: {os.path.basename(glb_path)}")
    try:
        figure = build_plotly_figure(glb_path)
    except Exception as exc:
        show_preview_mode_badge("Static")
        st.warning(f"3D preview failed to load interactively: {exc}")
        try:
            snapshot_bytes = build_static_snapshot(glb_path, str(exc))
            st.image(snapshot_bytes, caption="Static snapshot fallback", width="stretch")
        except Exception as snapshot_exc:
            st.info(f"Static snapshot also failed: {snapshot_exc}")
        return

    show_preview_mode_badge("Interactive")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False}, key=chart_key)


def persist_glb_in_session(glb_path):
    """Store GLB bytes in session so preview survives temp-file path loss."""
    if not glb_path or not os.path.exists(glb_path):
        return
    with open(glb_path, "rb") as f:
        glb_bytes = f.read()
    st.session_state["glb_path"] = glb_path
    st.session_state["glb_bytes"] = glb_bytes
    st.session_state["glb_name"] = os.path.basename(glb_path)
    update_snapshot_preview(glb_path)


def get_session_glb_path():
    """Return a valid on-disk GLB path from session, recreating temp file when needed."""
    glb_path = st.session_state.get("glb_path")
    if glb_path and os.path.exists(glb_path):
        st.session_state["glb_restored_from_bytes"] = False
        return glb_path

    glb_bytes = st.session_state.get("glb_bytes")
    if not glb_bytes:
        return ""

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
    tmp.write(glb_bytes)
    tmp.close()
    st.session_state["glb_path"] = tmp.name
    st.session_state["glb_restored_from_bytes"] = True
    if "glb_name" not in st.session_state:
        st.session_state["glb_name"] = os.path.basename(tmp.name)
    return tmp.name


def setup_pwa():
    """Setup PWA manifest and service worker."""
    pwa_html = """
    <script>
    const safeDocument = () => {
        try {
            if (window.parent && window.parent !== window && window.parent.document) {
                return window.parent.document;
            }
        } catch (err) {
            // parent is inaccessible due to cross-origin or sandbox restrictions
        }
        return document;
    };

    const ensureHeadTag = (doc, tagName, attrs) => {
        const selector = tagName + Object.entries(attrs)
            .map(([key, value]) => `[${key}="${value}"]`)
            .join('');

        if (!doc.querySelector(selector)) {
            const element = doc.createElement(tagName);
            Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
            doc.head.appendChild(element);
        }
    };

    const attachPwaMeta = () => {
        const doc = safeDocument();
        ensureHeadTag(doc, 'link', { rel: 'manifest', href: '/manifest.json' });
        ensureHeadTag(doc, 'meta', { name: 'theme-color', content: '#ff4b4b' });
        ensureHeadTag(doc, 'meta', { name: 'apple-mobile-web-app-capable', content: 'yes' });
        ensureHeadTag(doc, 'meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'default' });
        ensureHeadTag(doc, 'meta', { name: 'apple-mobile-web-app-title', content: '3D Generator' });
        ensureHeadTag(doc, 'link', {
            rel: 'apple-touch-icon',
            href: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTkyIiBoZWlnaHQ9IjE5MiIgdmlld0JveD0iMCAwIDE5MiAxOTIiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxOTIiIGhlaWdodD0iMTkyIiByeD0iMjQiIGZpbGw9IiNmZjRiNGIiLz4KPHBhdGggZD0iTTEyMCA2NGgzMnY2NGgtMzJ2LTY0ek0xMjggOTZ2MzJoLTE2di0zMmgxNnoiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo='
        });
    };

    const registerServiceWorker = () => {
        const targetWindow = (window.parent && window.parent !== window ? window.parent : window);
        if (targetWindow.navigator && 'serviceWorker' in targetWindow.navigator) {
            targetWindow.addEventListener('load', () => {
                targetWindow.navigator.serviceWorker.register('/sw.js', { updateViaCache: 'none' })
                    .then((registration) => {
                        registration.update();
                        console.log('ServiceWorker registration successful');
                    })
                    .catch(err => console.log('ServiceWorker registration failed:', err));
            });
        }
    };

    let deferredPrompt;
    const setupInstallPrompt = () => {
        const targetWindow = (window.parent && window.parent !== window ? window.parent : window);
        targetWindow.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            const installButton = document.getElementById('install-button');
            if (installButton) {
                installButton.style.display = 'block';
            }
        });
    };

    const bindInstallButton = () => {
        const installButton = document.getElementById('install-button');
        if (installButton) {
            installButton.addEventListener('click', async () => {
                if (!deferredPrompt) return;
                deferredPrompt.prompt();
                const { outcome } = await deferredPrompt.userChoice;
                deferredPrompt = null;
                installButton.style.display = 'none';
                console.log('Install prompt outcome:', outcome);
            });
        }
    };

    attachPwaMeta();
    registerServiceWorker();
    setupInstallPrompt();
    bindInstallButton();
    </script>
    """

    st.markdown(pwa_html, unsafe_allow_html=True)

def run_app():
    st.set_page_config(
        page_title="Image → 3D Generator",
        layout="wide",
        page_icon="🧊",
        initial_sidebar_state="collapsed",
    )

    setup_pwa()

    st.title("RKstudio → 🖼️ Image → 🧊 3D Model Generator")

    install_button_html = """
<div style="position: fixed; top: 10px; right: 10px; z-index: 1000;">
    <button id="install-button" style="
        background: #ff4b4b;
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
        display: none;
    ">
        📱 Install App
    </button>
</div>

<script>
const installButton = document.getElementById('install-button');

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    window.deferredPrompt = e;
    installButton.style.display = 'block';
});

installButton.addEventListener('click', async () => {
    const promptEvent = window.deferredPrompt;
    if (!promptEvent) return;

    promptEvent.prompt();
    const { outcome } = await promptEvent.userChoice;

    window.deferredPrompt = null;
    installButton.style.display = 'none';
});
</script>
"""

    st.markdown(install_button_html, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Image Generation", "3D Model Generation", "3D File Viewer"])

    with tab1:
        st.caption("Remember to check API usage!")
        mode = st.selectbox("Select Image Generation Mode", ["Fast_Mode_OA", "Slow_Mode_SDXL"], index=0, key="image_mode_select")
        prompt = st.text_area(
            "Image prompt",
            value="One Single Stylized simple multicoloured Common Holly performing Heterophylly whilst presented on a sturdy figurine base suitable for 3D printing.",
        )

        if st.button("Generate Image"):
            with st.spinner("Generating image..."):
                if mode == "Fast_Mode_OA":
                    image_path = generate_image_openai(prompt)
                else:
                    image_path = generate_image_sdxl(prompt)

                if image_path:
                    st.session_state["image_path"] = image_path
                    st.image(image_path, caption="Generated Image", width=400)

    with tab2:
        st.caption("Remember to check API usage!")
        mode = st.selectbox("Select 3D Model Generation Mode", ["Medium_Quality_Mode_STA", "High_Quality_Mode_TRO"], index=0, key="model_mode_select")
        image_path = st.session_state.get("image_path")

        uploaded = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(uploaded.read())
            tmp.close()
            image_path = tmp.name

        if not image_path:
            st.warning("Please generate or upload an image first.")
        else:
            st.image(image_path, caption="Input Image", width=300)

            if mode in ["Medium_Quality_Mode_STA", "Medium_Quality_Mode"]:
                texture_resolution = st.selectbox("Texture Resolution", ["512", "1024", "2048"], index=1)
                foreground_ratio = st.slider("Foreground Ratio", 0.1, 1.0, 0.85, 0.05)
                remesh = st.selectbox("Remesh", ["none", "quad", "triangle"])
                vertex_count = st.number_input("Vertex Count (-1 = auto)", value=-1)

                if st.button("Generate 3D Model", key="generate_3d_stability"):
                    with st.spinner("Generating 3D model..."):
                        glb_path = generate_3d_model_stability(
                            image_path,
                            texture_resolution,
                            foreground_ratio,
                            remesh,
                            vertex_count,
                        )

                        if glb_path:
                            persist_glb_in_session(glb_path)
                            st.success("3D model generated!")
                            with open(glb_path, "rb") as f:
                                st.download_button(
                                    "Download GLB",
                                    f,
                                    file_name="model.glb",
                                    mime="model/gltf-binary",
                                )
            elif mode in ["High_Quality_Mode_TRO", "High_Quality_Mode", "Tripo3D", "Fast_Mode"]:
                if st.button("Generate 3D Model", key="generate_3d_tripo"):
                    with st.spinner("Generating 3D model..."):
                        glb_path = generate_3d_model_tripo(image_path)

                        if glb_path:
                            persist_glb_in_session(glb_path)
                            st.success("3D model generated!")
                            with open(glb_path, "rb") as f:
                                st.download_button(
                                    "Download GLB",
                                    f,
                                    file_name="model.glb",
                                    mime="model/gltf-binary",
                                )

            current_glb_path = get_session_glb_path()
            if current_glb_path and os.path.exists(current_glb_path):
                snapshot_png = st.session_state.get("glb_preview_png")
                if snapshot_png:
                    st.subheader("Model Image Preview")
                    st.image(snapshot_png, caption="Snapshot of generated GLB", width="stretch")
                st.subheader("Current 3D Preview")
                restored_tag = "restored-from-session" if st.session_state.get("glb_restored_from_bytes") else "direct-file"
                st.caption(f"Debug render source: {current_glb_path} ({restored_tag})")
                display_3d_model(current_glb_path, chart_key="tab2_plotly_preview")
            elif st.session_state.get("glb_bytes"):
                st.info("GLB exists in session but could not be restored to disk for preview.")

    with tab3:
        st.caption("Preview an existing GLB file or inspect the most recently generated model.")
        viewer_upload = st.file_uploader("Upload a GLB file", type=["glb"], key="viewer_glb_upload")

        if viewer_upload:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
            tmp.write(viewer_upload.read())
            tmp.close()
            persist_glb_in_session(tmp.name)

        current_glb_path = get_session_glb_path()
        if current_glb_path and os.path.exists(current_glb_path):
            snapshot_png = st.session_state.get("glb_preview_png")
            if snapshot_png:
                st.subheader("Model Image Preview")
                st.image(snapshot_png, caption="Snapshot of selected GLB", width="stretch")
            restored_tag = "restored-from-session" if st.session_state.get("glb_restored_from_bytes") else "direct-file"
            st.caption(f"Debug render source: {current_glb_path} ({restored_tag})")
            display_3d_model(current_glb_path, chart_key="tab3_plotly_preview")
            with open(current_glb_path, "rb") as f:
                st.download_button(
                    "Download Current GLB",
                    f,
                    file_name=st.session_state.get("glb_name", os.path.basename(current_glb_path)),
                    mime="model/gltf-binary",
                    key="download_current_glb",
                )
        else:
            st.info("Upload a .glb file here or generate one in the 3D Model Generation tab.")


def get_openai_client():
    from openai import OpenAI

    api_key = get_api_key("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or secrets")
    return OpenAI(api_key=api_key)


def generate_image_openai(prompt: str) -> str:
    try:
        if not get_api_key("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment or secrets")

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
        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_API_KEY")
            or os.environ.get("RKStudioHF1")
        )
        if not hf_token:
            raise ValueError("Set HF_TOKEN, HUGGINGFACE_API_KEY, or RKStudioHF1 in your .env for SDXL mode")

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
        api_key = os.environ.get("TRIPO3D_API_KEY") or os.environ.get("RKStudioTripo")
        if not api_key:
            raise ValueError("Set TRIPO3D_API_KEY or RKStudioTripo in your .env")

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


if __name__ == "__main__":
    run_app()