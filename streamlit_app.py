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


def get_cohere_api_key():
    """Get Cohere API key from common env/secret variable names."""
    candidate_names = [
        "COHERE_API_KEY",
        "COHERE_KEY",
        "RKStudioCohere",
        "RKSTUDIO_COHERE_API_KEY",
    ]
    for name in candidate_names:
        value = get_api_key(name)
        if value:
            return value
    return None


def get_cohere_response(api_key, user_input, history, model_name, temperature, max_tokens):
    """Generate a Cohere response, supporting both V1 and V2 SDK clients."""
    import cohere

    # Try V1 first.
    client = cohere.Client(api_key)
    v1_history = [
        {
            "role": "USER" if msg["role"] == "user" else "CHATBOT",
            "message": msg["content"],
        }
        for msg in history
    ]
    try:
        response = client.chat(
            message=user_input,
            chat_history=v1_history,
            model=model_name,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        text = getattr(response, "text", "")
        if text:
            return text.strip()
    except Exception:
        pass

    # Fallback to V2.
    client_v2 = cohere.ClientV2(api_key)
    messages = [
        {
            "role": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"],
        }
        for msg in history
    ]
    messages.append({"role": "user", "content": user_input})
    response_v2 = client_v2.chat(
        model=model_name,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )

    if hasattr(response_v2, "message") and getattr(response_v2.message, "content", None):
        parts = []
        for block in response_v2.message.content:
            if getattr(block, "type", "") == "text":
                parts.append(getattr(block, "text", ""))
        text = "\n".join([part for part in parts if part]).strip()
        if text:
            return text

    raise ValueError("No response text returned by Cohere")


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
    st.session_state.pop("conversion_obj_bytes", None)
    st.session_state.pop("conversion_stl_bytes", None)
    st.session_state.pop("conversion_source_name", None)
    update_snapshot_preview(glb_path)


def load_glb_as_mesh(glb_path):
    """Load a GLB and normalize it to a single mesh for format export."""
    loaded = trimesh.load(glb_path, force="scene")

    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        scene_meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not scene_meshes:
            raise ValueError("No mesh geometry found in GLB file")
        mesh = trimesh.util.concatenate(scene_meshes) if len(scene_meshes) > 1 else scene_meshes[0].copy()
    else:
        raise ValueError("Unsupported GLB content")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("GLB mesh has no faces to export")

    return mesh


def export_mesh_bytes(mesh, file_type):
    """Export a trimesh mesh to bytes for Streamlit download buttons."""
    payload = mesh.export(file_type=file_type)
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if isinstance(payload, bytes):
        return payload
    raise ValueError(f"Unsupported export payload for {file_type}")


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

    tab_prompt, tab_chat, tab1, tab2, tab3, tab4_obj, tab4_stl, tab_blog, tab_links = st.tabs([
        "3D Model Prompt Ideas (Cohere)",
        "AI Chat Assistant",
        "Image Generation",
        "3D Model Generation",
        "3D File Viewer",
        "Convert to OBJ",
        "Convert to STL",
        "Blog Writer (Cohere)",
        "Links",
    ])

    def sanitize_filename(name):
        import re

        return re.sub(r"[^\w\-]+", "_", (name or "").strip())

    with tab1:
        st.caption("Remember to check API usage!")
        mode = st.selectbox("Select Image Generation Mode", ["Fast_Mode_OA", "Slow_Mode_SDXL"], index=0, key="image_mode_select")
        prompt = st.text_area(
            "Image prompt",
            value="One Single Stylized simple multicoloured Common Holly performing Heterophylly whilst presented on a sturdy figurine base suitable for 3D printing.",
        )

        if st.button("Generate Image"):
            with st.spinner("Generating image..."):
                image_path = generate_image_openai(prompt) if mode == "Fast_Mode_OA" else generate_image_sdxl(prompt)
                if image_path:
                    st.session_state["image_path"] = image_path
                    st.image(image_path, caption="Generated Image", width=400)

    with tab2:
        st.caption("Remember to check API usage!")
        mode = st.selectbox("Select 3D Model Generation Mode", ["Medium_Quality_Mode_STA", "High_Quality_Mode_TRO"], index=0, key="model_mode_select")
        image_path = st.session_state.get("image_path")

        uploaded = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"], key="model_image_upload")
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
                filename_glb = st.text_input("GLB filename (no extension)", value="model", key="filename_glb_tab2_sta")

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
                            safe_name = sanitize_filename(filename_glb) or "model"
                            with open(glb_path, "rb") as f:
                                st.download_button(
                                    "Download GLB",
                                    f,
                                    file_name=f"{safe_name}.glb",
                                    mime="model/gltf-binary",
                                )
            else:
                filename_glb = st.text_input("GLB filename (no extension)", value="model", key="filename_glb_tab2_tripo")
                if st.button("Generate 3D Model", key="generate_3d_tripo"):
                    with st.spinner("Generating 3D model..."):
                        glb_path = generate_3d_model_tripo(image_path)
                        if glb_path:
                            persist_glb_in_session(glb_path)
                            st.success("3D model generated!")
                            safe_name = sanitize_filename(filename_glb) or "model"
                            with open(glb_path, "rb") as f:
                                st.download_button(
                                    "Download GLB",
                                    f,
                                    file_name=f"{safe_name}.glb",
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
            default_glb_name = os.path.splitext(st.session_state.get("glb_name", os.path.basename(current_glb_path)))[0]
            filename_glb_view = st.text_input("GLB filename (no extension)", value=default_glb_name, key="filename_glb_tab3")
            safe_name = sanitize_filename(filename_glb_view) or "model"
            with open(current_glb_path, "rb") as f:
                st.download_button(
                    "Download Current GLB",
                    f,
                    file_name=f"{safe_name}.glb",
                    mime="model/gltf-binary",
                    key="download_current_glb",
                )
        else:
            st.info("Upload a .glb file here or generate one in the 3D Model Generation tab.")

    with tab4_obj:
        st.caption("Convert a GLB to OBJ and choose the output filename.")
        obj_upload = st.file_uploader("Upload a GLB for OBJ conversion (optional)", type=["glb"], key="obj_conversion_glb_upload")

        if obj_upload:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
            tmp.write(obj_upload.read())
            tmp.close()
            st.session_state["obj_glb_path"] = tmp.name

        obj_glb_path = st.session_state.get("obj_glb_path") or get_session_glb_path()
        if obj_glb_path and os.path.exists(obj_glb_path):
            source_name = os.path.splitext(os.path.basename(obj_glb_path))[0]
            st.write(f"Current GLB source: {os.path.basename(obj_glb_path)}")
            filename_obj = st.text_input("OBJ filename (no extension)", value=source_name, key="filename_obj_tab")

            if st.button("Convert to OBJ", key="convert_to_obj"):
                try:
                    mesh = load_glb_as_mesh(obj_glb_path)
                    st.session_state["obj_bytes"] = export_mesh_bytes(mesh, "obj")
                    st.success("Conversion to OBJ complete.")
                except Exception as exc:
                    st.error(f"OBJ conversion failed: {exc}")

            obj_bytes = st.session_state.get("obj_bytes")
            if obj_bytes:
                safe_name_obj = sanitize_filename(filename_obj) or "model"
                st.download_button(
                    "Download OBJ",
                    obj_bytes,
                    file_name=f"{safe_name_obj}.obj",
                    mime="model/obj",
                    key="download_obj",
                )
        else:
            st.info("Upload a .glb file here, or generate one in the 3D Model Generation tab.")

    with tab4_stl:
        st.caption("Convert a GLB to STL and choose the output filename.")
        stl_upload = st.file_uploader("Upload a GLB for STL conversion (optional)", type=["glb"], key="stl_conversion_glb_upload")

        if stl_upload:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
            tmp.write(stl_upload.read())
            tmp.close()
            st.session_state["stl_glb_path"] = tmp.name

        stl_glb_path = st.session_state.get("stl_glb_path") or get_session_glb_path()
        if stl_glb_path and os.path.exists(stl_glb_path):
            source_name = os.path.splitext(os.path.basename(stl_glb_path))[0]
            st.write(f"Current GLB source: {os.path.basename(stl_glb_path)}")
            filename_stl = st.text_input("STL filename (no extension)", value=source_name, key="filename_stl_tab")

            if st.button("Convert to STL", key="convert_to_stl"):
                try:
                    mesh = load_glb_as_mesh(stl_glb_path)
                    st.session_state["stl_bytes"] = export_mesh_bytes(mesh, "stl")
                    st.success("Conversion to STL complete.")
                except Exception as exc:
                    st.error(f"STL conversion failed: {exc}")

            stl_bytes = st.session_state.get("stl_bytes")
            if stl_bytes:
                safe_name_stl = sanitize_filename(filename_stl) or "model"
                st.download_button(
                    "Download STL",
                    stl_bytes,
                    file_name=f"{safe_name_stl}.stl",
                    mime="model/stl",
                    key="download_stl",
                )
        else:
            st.info("Upload a .glb file here, or generate one in the 3D Model Generation tab.")

    with tab_chat:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:0.45rem;margin-bottom:0.15rem;">
                <span style="display:inline-flex;align-items:center;justify-content:center;width:1.25rem;height:1.25rem;border-radius:0.35rem;background:#e0f2fe;border:1px solid #bae6fd;overflow:hidden;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                        <rect x="4" y="7" width="16" height="12" rx="3" stroke="#0369a1" stroke-width="1.7"/>
                        <circle cx="9" cy="13" r="1.4" fill="#0284c7"/>
                        <circle cx="15" cy="13" r="1.4" fill="#0284c7"/>
                        <rect x="10" y="16.2" width="4" height="1.6" rx="0.8" fill="#0284c7"/>
                        <path d="M12 3.8V7" stroke="#0369a1" stroke-width="1.7" stroke-linecap="round"/>
                        <circle cx="12" cy="3" r="1.2" fill="#0369a1"/>
                    </svg>
                </span>
                <span style="font-size:0.92rem;color:#475569;">Ask the assistant for image prompts, styling ideas, and model concepts.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cohere_api_key = get_cohere_api_key()
        if cohere_api_key:
            st.success("Cohere key detected. Assistant is ready.")
        else:
            st.warning("Cohere key not found in .env or Streamlit secrets.")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "chat_input" not in st.session_state:
            st.session_state["chat_input"] = ""
        if "chat_input_next" not in st.session_state:
            st.session_state["chat_input_next"] = None
        if "cohere_model" not in st.session_state:
            st.session_state["cohere_model"] = "command-r-plus-08-2024"
        if "cohere_temperature" not in st.session_state:
            st.session_state["cohere_temperature"] = 0.7
        if "cohere_max_tokens" not in st.session_state:
            st.session_state["cohere_max_tokens"] = 450

        # Apply pending chat input updates before the widget is instantiated.
        if st.session_state["chat_input_next"] is not None:
            st.session_state["chat_input"] = st.session_state["chat_input_next"]
            st.session_state["chat_input_next"] = None

        with st.expander("Chat Settings", expanded=True):
            st.selectbox(
                "Cohere model",
                [
                    "command-r-plus-08-2024",
                    "command-a-03-2025",
                    "command-r7b-12-2024",
                ],
                key="cohere_model",
            )
            st.slider("Temperature", 0.0, 1.0, key="cohere_temperature", step=0.05)
            st.slider("Max output tokens", 100, 2000, key="cohere_max_tokens", step=50)

        cols = st.columns(2)
        with cols[0]:
            if st.button("Use starter question", key="starter_question"):
                st.session_state["chat_input_next"] = "Give me 5 creative image ideas that convert well into printable 3D figurines."
                st.rerun()
        with cols[1]:
            if st.button("Clear chat", key="clear_chat"):
                st.session_state["chat_history"] = []
                st.session_state["chat_input_next"] = ""
                st.rerun()

        user_input = st.text_input(
            "Ask the assistant anything about image + 3D concept ideas",
            key="chat_input",
            placeholder="Example: Suggest a stylized creature concept with clean silhouette for 3D conversion.",
        )
        if st.button("Send", key="send_chat"):
            if not user_input.strip():
                st.info("Type a prompt first.")
            elif not cohere_api_key:
                st.error("Set a Cohere API key in .env (COHERE_API_KEY is recommended).")
            else:
                try:
                    history = st.session_state["chat_history"]
                    selected_model = st.session_state.get("cohere_model", "command-a-03-2025")
                    temperature = st.session_state.get("cohere_temperature", 0.7)
                    max_tokens = st.session_state.get("cohere_max_tokens", 450)
                    model_candidates = [
                        selected_model,
                        "command-r-plus-08-2024",
                        "command-a-03-2025",
                        "command-r7b-12-2024",
                    ]
                    # Preserve order while deduplicating.
                    model_candidates = list(dict.fromkeys(model_candidates))

                    last_error = None
                    answer = ""
                    for model_name in model_candidates:
                        try:
                            with st.spinner("Assistant is thinking..."):
                                answer = get_cohere_response(
                                    api_key=cohere_api_key,
                                    user_input=user_input,
                                    history=history,
                                    model_name=model_name,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                )
                            if answer:
                                st.session_state["cohere_model"] = model_name
                                break
                        except Exception as exc:
                            last_error = exc

                    if not answer and last_error:
                        raise last_error

                    if not answer:
                        answer = "I could not generate a response. Please try again."

                    st.session_state["chat_history"].append({"role": "user", "content": user_input})
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                    st.session_state["chat_input_next"] = ""
                    st.rerun()
                except Exception as exc:
                    st.error(f"Cohere chat failed: {exc}")

        for msg in st.session_state["chat_history"]:
            label = "You" if msg["role"] == "user" else "Assistant"
            st.markdown(f"**{label}:** {msg['content']}")

    with tab_blog:
        st.subheader("Cohere Blog Writer")
        st.caption("Ask Cohere a question and generate a structured blog draft in Markdown.")

        cohere_api_key = get_cohere_api_key()
        if cohere_api_key:
            st.success("Cohere key detected. Blog writer is ready.")
        else:
            st.warning("Cohere key not found in .env or Streamlit secrets.")

        if "blog_question" not in st.session_state:
            st.session_state["blog_question"] = "How can small studios use AI to speed up 3D concept workflows?"
        if "blog_audience" not in st.session_state:
            st.session_state["blog_audience"] = "Indie creators and hobby makers"
        if "blog_tone" not in st.session_state:
            st.session_state["blog_tone"] = "Practical and friendly"
        if "blog_length" not in st.session_state:
            st.session_state["blog_length"] = 900
        if "blog_include_outline" not in st.session_state:
            st.session_state["blog_include_outline"] = True
        if "blog_output" not in st.session_state:
            st.session_state["blog_output"] = ""
        if "blog_model" not in st.session_state:
            st.session_state["blog_model"] = "command-r-plus-08-2024"

        st.selectbox(
            "Cohere model",
            [
                "command-r-plus-08-2024",
                "command-a-03-2025",
                "command-r7b-12-2024",
            ],
            key="blog_model",
        )
        st.text_area(
            "Question or topic for the blog",
            key="blog_question",
            placeholder="Example: What are the best ways to create 3D-printable character concepts from text prompts?",
            height=100,
        )
        st.text_input("Target audience", key="blog_audience")
        st.text_input("Tone", key="blog_tone")
        st.slider("Target word count", min_value=300, max_value=2000, step=100, key="blog_length")
        st.checkbox("Include a short outline at the top", key="blog_include_outline")

        if st.button("Generate Blog Draft", key="generate_blog_draft"):
            if not st.session_state["blog_question"].strip():
                st.info("Add a question or topic first.")
            elif not cohere_api_key:
                st.error("Set a Cohere API key in .env (COHERE_API_KEY is recommended).")
            else:
                try:
                    include_outline = "yes" if st.session_state.get("blog_include_outline") else "no"
                    user_prompt = (
                        "You are a senior technical blog writer. Write a clear, engaging Markdown blog post.\\n"
                        f"Primary question/topic: {st.session_state.get('blog_question', '').strip()}\\n"
                        f"Target audience: {st.session_state.get('blog_audience', '').strip()}\\n"
                        f"Tone: {st.session_state.get('blog_tone', '').strip()}\\n"
                        f"Target length: about {int(st.session_state.get('blog_length', 900))} words\\n"
                        f"Include a short outline first: {include_outline}\\n\\n"
                        "Return Markdown only with this structure:\\n"
                        "1) Title\\n"
                        "2) (Optional) Outline\\n"
                        "3) Introduction\\n"
                        "4) Main sections with H2/H3 headings\\n"
                        "5) Conclusion with practical next steps\\n"
                        "Keep the content actionable and include concrete examples where useful."
                    )

                    with st.spinner("Generating blog draft with Cohere..."):
                        blog_text = get_cohere_response(
                            api_key=cohere_api_key,
                            user_input=user_prompt,
                            history=[],
                            model_name=st.session_state.get("blog_model", "command-r-plus-08-2024"),
                            temperature=0.6,
                            max_tokens=1800,
                        )

                    st.session_state["blog_output"] = blog_text
                    st.success("Blog draft generated.")
                except Exception as exc:
                    st.error(f"Blog generation failed: {exc}")

        blog_output = st.session_state.get("blog_output", "")
        if blog_output:
            st.markdown("### Draft Preview")
            st.markdown(blog_output)
            blog_name = sanitize_filename(st.session_state.get("blog_question", "blog-draft")[:60]) or "blog-draft"
            st.download_button(
                "Download Blog Markdown",
                data=blog_output,
                file_name=f"{blog_name}.md",
                mime="text/markdown",
                key="download_blog_markdown",
            )

    with tab_prompt:
        st.subheader("3D Image Prompt Ideas (Cohere)")
        st.caption("Enter a topic and a visual style to generate 3D image prompt ideas.")

        cohere_api_key = get_cohere_api_key()
        if cohere_api_key:
            st.success("Cohere key detected. Prompt ideas are ready.")
        else:
            st.warning("Cohere key not found in .env or Streamlit secrets.")

        _STYLE_PRESETS = {
            "As Simple As Possible": (
                "extremely simple and minimal style optimised for clean 3D model conversion",
                "single object centered on a plain white background, basic geometric shapes only, no fine detail, no accessories, no background elements, solid flat colors, even neutral lighting, nothing that would confuse a 3D mesh generator",
            ),
            "Simple & Cartoony": (
                "simple, playful cartoon style",
                "rounded shapes, smooth surfaces, toy-like proportions, bright but limited colors, soft even lighting, no complex details",
            ),
            "Realistic & Detailed": (
                "photorealistic and highly detailed style",
                "accurate proportions, fine surface detail, realistic materials and textures, natural lighting with shadows",
            ),
            "Low-poly / Stylized": (
                "low-poly stylized style",
                "flat shaded geometric facets, minimal color palette, clean silhouette, isometric-friendly proportions",
            ),
            "Sci-fi / Futuristic": (
                "sci-fi futuristic style",
                "sleek hard-surface forms, metallic and glowing materials, neon accent colors, dramatic rim lighting",
            ),
            "Hand-crafted / Clay": (
                "hand-crafted clay or stop-motion style",
                "slightly imperfect organic shapes, matte clay-like surfaces, warm studio lighting, tactile texture cues",
            ),
        }

        if "prompt_topic" not in st.session_state:
            st.session_state["prompt_topic"] = ""
        if "prompt_examples_count" not in st.session_state:
            st.session_state["prompt_examples_count"] = 5
        if "prompt_examples_audience" not in st.session_state:
            st.session_state["prompt_examples_audience"] = "3D artists, illustrators, and hobby makers"
        if "prompt_style_preset" not in st.session_state:
            st.session_state["prompt_style_preset"] = "As Simple As Possible"
        if "prompt_style_custom" not in st.session_state:
            st.session_state["prompt_style_custom"] = ""
        if "prompt_examples_output" not in st.session_state:
            st.session_state["prompt_examples_output"] = ""
        if "prompt_examples_model" not in st.session_state:
            st.session_state["prompt_examples_model"] = "command-r-plus-08-2024"

        st.selectbox(
            "Cohere model",
            [
                "command-r-plus-08-2024",
                "command-a-03-2025",
                "command-r7b-12-2024",
            ],
            key="prompt_examples_model",
        )
        st.text_area(
            "Topic for 3D image ideas",
            key="prompt_topic",
            placeholder="Example: a friendly robot baker with rounded shapes and bright colors",
            height=100,
        )
        st.selectbox(
            "Visual style",
            list(_STYLE_PRESETS.keys()) + ["Custom…"],
            key="prompt_style_preset",
        )
        if st.session_state.get("prompt_style_preset") == "Custom…":
            st.text_input(
                "Describe your custom style",
                key="prompt_style_custom",
                placeholder="Example: dark fantasy with mossy stone textures and dramatic shadows",
            )
        st.text_input("Intended use or audience", key="prompt_examples_audience")
        st.slider("Number of examples", min_value=3, max_value=10, step=1, key="prompt_examples_count")

        if st.button("Generate 3D Prompt Ideas", key="generate_prompt_examples"):
            if not st.session_state["prompt_topic"].strip():
                st.info("Add a topic first.")
            elif not cohere_api_key:
                st.error("Set a Cohere API key in .env (COHERE_API_KEY is recommended).")
            else:
                try:
                    chosen_preset = st.session_state.get("prompt_style_preset", "Simple & Cartoony")
                    if chosen_preset == "Custom…":
                        custom_style = st.session_state.get("prompt_style_custom", "").strip()
                        if not custom_style:
                            st.info("Describe your custom style above first.")
                            st.stop()
                        style_desc = custom_style
                        style_constraints = custom_style
                        display_name = "Custom"
                    else:
                        style_desc, style_constraints = _STYLE_PRESETS.get(
                            chosen_preset, _STYLE_PRESETS["Simple & Cartoony"]
                        )
                        display_name = chosen_preset
                    user_prompt = (
                        f"You are an expert 3D image prompt engineer. Generate prompt ideas for creating 3D images in a {style_desc}.\n"
                        f"Topic: {st.session_state.get('prompt_topic', '').strip()}\n"
                        f"Intended use or audience: {st.session_state.get('prompt_examples_audience', '').strip()}\n"
                        f"Visual style: {display_name}\n"
                        f"Number of examples: {int(st.session_state.get('prompt_examples_count', 5))}\n\n"
                        "Return Markdown only with this structure:\n"
                        "1) A short title for each example\n"
                        "2) A detailed prompt inside a fenced code block\n"
                        "3) One brief sentence explaining what kind of 3D image it is best for\n\n"
                        f"Tailor every prompt to the chosen style. Key visual cues to include: {style_constraints}."
                    )

                    with st.spinner(f"Generating {display_name} 3D prompt ideas with Cohere..."):
                        prompt_examples_text = get_cohere_response(
                            api_key=cohere_api_key,
                            user_input=user_prompt,
                            history=[],
                            model_name=st.session_state.get("prompt_examples_model", "command-r-plus-08-2024"),
                            temperature=0.75,
                            max_tokens=1800,
                        )

                    st.session_state["prompt_examples_output"] = prompt_examples_text
                    st.success(f"{display_name} 3D prompt ideas generated.")
                except Exception as exc:
                    st.error(f"3D prompt generation failed: {exc}")

        prompt_examples_output = st.session_state.get("prompt_examples_output", "")
        if prompt_examples_output:
            chosen_preset = st.session_state.get("prompt_style_preset", "Simple & Cartoony")
            display_name = "Custom" if chosen_preset == "Custom…" else chosen_preset
            st.markdown(f"### Generated {display_name} 3D Prompt Ideas")
            st.markdown(prompt_examples_output)
            prompt_name = sanitize_filename(st.session_state.get("prompt_topic", "prompt-ideas")[:60]) or "prompt-ideas"
            st.download_button(
                "Download Prompt Ideas Markdown",
                data=prompt_examples_output,
                file_name=f"{prompt_name}.md",
                mime="text/markdown",
                key="download_prompt_examples_markdown",
            )

    with tab_links:
        st.subheader("Links")
        st.caption("Quick links to tools for writing, ecommerce, and 3D creation.")

        links = [
            ("Canva", "https://www.canva.com"),
            ("Medium", "https://medium.com"),
            ("ChatGPT", "https://chatgpt.com"),
            ("Claude", "https://claude.ai"),
            ("Shopify", "https://www.shopify.com"),
            ("TikTok Shop", "https://seller-uk.tiktok.com/homepage?shop_region=GB"),
            ("TikTok", "https://www.tiktok.com/"),
            ("Lark", "https://www.larksuite.com/en_sg/"),
            ("Printful", "https://www.printful.com"),
            ("Etsy", "https://www.etsy.com"),
            ("Tinkercad", "https://www.tinkercad.com"),
            ("Cults3D", "https://cults3d.com"),
        ]

        cols = st.columns(3)
        for idx, (label, url) in enumerate(links):
            with cols[idx % 3]:
                if hasattr(st, "link_button"):
                    st.link_button(label, url, use_container_width=True)
                else:
                    st.markdown(f"[{label}]({url})")

        st.markdown("### API Links")
        st.caption("Direct access to API platforms and docs.")
        api_links = [
            ("Stability API", "https://platform.stability.ai"),
            ("OpenAI API", "https://platform.openai.com"),
            ("Cohere API", "https://docs.cohere.com"),
        ]

        api_cols = st.columns(3)
        for idx, (label, url) in enumerate(api_links):
            with api_cols[idx % 3]:
                if hasattr(st, "link_button"):
                    st.link_button(label, url, use_container_width=True)
                else:
                    st.markdown(f"[{label}]({url})")


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