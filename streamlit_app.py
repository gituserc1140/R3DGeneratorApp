import os
import tempfile
import traceback
import io
import math
import base64
import time
from urllib.parse import quote

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
import trimesh
from dotenv import load_dotenv
from PIL import Image, ImageDraw


load_dotenv()


USER_KEY_FIELDS = [
    ("user_openai_key", "OpenAI API key", ["OPENAI_API_KEY"]),
    ("user_cohere_key", "Cohere API key", ["COHERE_API_KEY", "COHERE_KEY", "RKStudioCohere", "RKSTUDIO_COHERE_API_KEY"]),
    ("user_stability_key", "Stability AI API key", ["STABILITY_KEY", "STABILITY_API_KEY"]),
    ("user_openrouter_key", "OpenRouter API key", ["OPENROUTER_API_KEY"]),
    ("user_together_key", "Together AI API key", ["TOGETHER_API_KEY"]),
    ("user_replicate_key", "Replicate API key", ["REPLICATE_API_KEY"]),
    ("user_pollinations_key", "Pollinations API key (optional)", ["POLLINATIONS_API_KEY", "POLLINATIONS_KEY"]),
]


def get_user_api_key(key_name):
    """Return a key the end user entered in the sidebar, if any."""
    try:
        for state_key, _label, names in USER_KEY_FIELDS:
            if key_name in names:
                value = (st.session_state.get(state_key) or "").strip()
                if value:
                    return value
    except Exception:
        pass
    return None


def render_api_key_inputs():
    """Sidebar inputs so end users can supply their own API keys (session only)."""
    with st.sidebar:
        st.header("🔑 Your API keys")
        st.caption("Optional. Keys are kept only in this browser session and override server-configured keys.")
        for state_key, label, _names in USER_KEY_FIELDS:
            value = st.text_input(label, type="password", key=state_key)
            if (value or "").strip():
                st.caption(":green[✅ Key accepted for this session]")


def get_api_key(key_name):
    """Get API key from the user's session, environment, or Streamlit secrets."""
    user_value = get_user_api_key(key_name)
    if user_value:
        return user_value
    # Re-read local .env so rotated keys are picked up without a full app restart.
    load_dotenv(override=True)
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


def _provider_options():
    """List the image generation providers exposed in the UI."""
    return [
        "OpenAI",
        "Pollinations",
        "OpenRouter",
        "Together AI",
        "Replicate",
    ]


def _get_image_provider_name():
    """Return the selected image provider from session or env."""
    try:
        value = (st.session_state.get("image_provider") or "").strip()
        if value:
            return value
    except Exception:
        pass
    return os.environ.get("DEFAULT_IMAGE_PROVIDER", "OpenAI")


def _require_key_for_provider(provider):
    """Return the env variable name required for a provider or raise a friendly error."""
    keys = {
        "OpenAI": "OPENAI_API_KEY",
        "OpenRouter": "OPENROUTER_API_KEY",
        "Together AI": "TOGETHER_API_KEY",
        "Replicate": "REPLICATE_API_KEY",
        "Pollinations": "POLLINATIONS_API_KEY",
    }
    key_name = keys.get(provider)
    if key_name is None:
        return None
    value = get_api_key(key_name)
    if value:
        return value
    raise ValueError(f"{key_name} not found in environment or secrets")


def _fetch_image_bytes_from_url(url, timeout=90):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def _write_temp_image(data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(data)
    tmp.close()
    return tmp.name


def _generate_image_pollinations(prompt: str) -> str:
    """Generate an image via Pollinations with a simple URL-based API."""
    api_key = get_api_key("POLLINATIONS_API_KEY") or get_api_key("POLLINATIONS_KEY")
    params = {
        "prompt": prompt,
        "model": "flux",
        "seed": int(time.time() * 1000) % 100000,
        "width": 1024,
        "height": 1024,
        "nologo": "true",
    }
    if api_key:
        params["token"] = api_key
    url = "https://image.pollinations.ai/prompt/" + quote(prompt)
    response = requests.get(url, params=params, timeout=120)
    if response.status_code >= 400:
        raise RuntimeError(f"Pollinations image request failed ({response.status_code}): {response.text[:400]}")
    if response.headers.get("content-type", "").startswith("image/"):
        return _write_temp_image(response.content)
    raise RuntimeError("Pollinations did not return image data.")


def _generate_image_openrouter(prompt: str) -> str:
    """Generate an image via OpenRouter's image-capable models."""
    api_key = get_api_key("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment or secrets")

    payload = {
        "model": os.environ.get("OPENROUTER_IMAGE_MODEL", "openai/gpt-image-1"),
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
        "response_format": {"type": "image_url"},
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer " + api_key,
            "HTTP-Referer": "https://studio-rk3dps-community-app.streamlit.app/",
            "X-Title": "StudioRK3dps",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenRouter image request failed ({response.status_code}): {response.text[:500]}")
    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                url = item.get("image_url", {}).get("url") or item.get("url")
                if url:
                    return _write_temp_image(_fetch_image_bytes_from_url(url))
    if isinstance(content, str):
        if content.startswith("http"):
            return _write_temp_image(_fetch_image_bytes_from_url(content))
        raise RuntimeError("OpenRouter returned a non-image response.")
    raise RuntimeError("OpenRouter response did not include image data.")


def _generate_image_together(prompt: str) -> str:
    """Generate an image via Together AI image models."""
    api_key = get_api_key("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not found in environment or secrets")

    payload = {
        "model": os.environ.get("TOGETHER_IMAGE_MODEL", "blackforestlabs/flux-1.1-pro"),
        "prompt": prompt,
        "height": 1024,
        "width": 1024,
        "steps": 28,
        "n": 1,
    }
    response = requests.post(
        "https://api.together.xyz/v1/images/generations",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Together AI image request failed ({response.status_code}): {response.text[:500]}")
    payload = response.json()
    image_url = payload.get("data", [{}])[0].get("url") or payload.get("output", [{}])[0]
    if not image_url:
        raise RuntimeError("Together AI response did not include an image URL.")
    return _write_temp_image(_fetch_image_bytes_from_url(image_url))


def _generate_image_replicate(prompt: str) -> str:
    """Generate an image via Replicate model predictions."""
    api_key = get_api_key("REPLICATE_API_KEY")
    if not api_key:
        raise ValueError("REPLICATE_API_KEY not found in environment or secrets")

    model_name = os.environ.get("REPLICATE_IMAGE_MODEL", "blackforestlabs/flux-1.1-pro")
    payload = {
        "version": os.environ.get("REPLICATE_IMAGE_VERSION") or None,
        "input": {"prompt": prompt, "width": 1024, "height": 1024},
    }
    if payload["version"] is None:
        payload.pop("version")

    response = requests.post(
        f"https://api.replicate.com/v1/models/{quote(model_name)}/predictions",
        headers={
            "Authorization": "Token " + api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Replicate image request failed ({response.status_code}): {response.text[:500]}")
    result = response.json()
    prediction_id = result.get("id")
    if not prediction_id:
        raise RuntimeError("Replicate response did not include a prediction ID.")

    status_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
    for _ in range(90):
        status = requests.get(status_url, headers={"Authorization": "Token " + api_key}, timeout=60)
        status.raise_for_status()
        status_json = status.json()
        output = status_json.get("output")
        if isinstance(output, list) and output:
            image_url = output[0]
            return _write_temp_image(_fetch_image_bytes_from_url(image_url))
        if status_json.get("status") == "succeeded" and isinstance(output, str):
            return _write_temp_image(_fetch_image_bytes_from_url(output))
        time.sleep(2)
    raise RuntimeError("Replicate image generation timed out.")


def generate_image_with_provider(prompt: str, provider: str) -> str:
    """Generate an image using any configured provider."""
    provider = provider or "OpenAI"
    if provider == "OpenAI":
        return generate_image_openai(prompt)
    if provider == "Pollinations":
        return _generate_image_pollinations(prompt)
    if provider == "OpenRouter":
        return _generate_image_openrouter(prompt)
    if provider == "Together AI":
        return _generate_image_together(prompt)
    if provider == "Replicate":
        return _generate_image_replicate(prompt)
    raise ValueError(f"Unsupported image provider: {provider}")


def _project_point_iso(point):
    """Project a 3D point into a 2D isometric-like view."""
    x, y, z = point
    cos_30 = math.sqrt(3) / 2
    sin_30 = 0.5
    return ((x - y) * cos_30, (x + y) * sin_30 - z)


def build_static_snapshot(glb_path, note=""):
    """Render a static PNG snapshot from mesh vertices when triangulation fails."""
    width, height = 960, 640
    image = Image.new("RGB", (width, height), "#ffffff")
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
    image = Image.new("RGB", (width, height), "#ffffff")
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
        "Physical": ("#991b1b", "#fee2e2", "#fecaca"),
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


def get_default_viewer_settings():
    """Return default interactive viewer settings."""
    return {
        "max_faces": 60000,
        "normalize": False,
        "wireframe": False,
        "wireframe_limit": 12000,
        "ambient": 0.44,
        "diffuse": 0.48,
        "specular": 0.04,
        "roughness": 0.84,
        "flatshading": False,
        "bg_color": "#ffffff",
        "camera_preset": "Isometric",
        "projection": "perspective",
        "show_axes": False,
    }


def _camera_eye_from_preset(name):
    """Map a friendly camera preset name to a Plotly camera eye vector."""
    presets = {
        "Isometric": {"x": 1.65, "y": 1.65, "z": 1.0},
        "Front": {"x": 0.0, "y": 2.5, "z": 0.2},
        "Top": {"x": 0.01, "y": 0.01, "z": 2.8},
        "Left": {"x": -2.5, "y": 0.1, "z": 0.2},
    }
    return presets.get(name, presets["Isometric"])


def _downsample_faces(faces, max_faces):
    """Reduce face count deterministically for smoother browser rendering."""
    if len(faces) <= max_faces:
        return faces
    step = max(1, int(math.ceil(len(faces) / max_faces)))
    sampled = faces[::step]
    if len(sampled) > max_faces:
        sampled = sampled[:max_faces]
    return sampled


def _normalize_vertices(vertices):
    """Center and scale vertices to keep camera framing consistent across models."""
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = float((maxs - mins).max())
    if extent <= 1e-9:
        extent = 1.0
    return (vertices - center) / extent


def _to_plotly_rgba_colors(colors):
    """Convert trimesh color arrays into Plotly-friendly rgba strings."""
    if colors is None:
        return None

    array = np.asarray(colors)
    if array.ndim != 2 or array.shape[1] < 3 or len(array) == 0:
        return None

    rgb = array[:, :3].astype(float)
    if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(rgb)) <= 1.0:
        rgb = np.clip(rgb * 255.0, 0, 255)

    if array.shape[1] >= 4:
        alpha = array[:, 3].astype(float)
        if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(alpha)) <= 1.0:
            alpha = np.clip(alpha * 255.0, 0, 255)
    else:
        alpha = np.full((len(array),), 255.0)

    return [
        f"rgba({int(r)},{int(g)},{int(b)},{float(a) / 255.0:.3f})"
        for (r, g, b), a in zip(rgb, alpha)
    ]


def _build_wireframe_trace(vertices, faces, edge_limit):
    """Build an optional wireframe overlay trace with bounded edge count."""
    edges = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (int(u), int(v)) if u <= v else (int(v), int(u))
            if edge not in edges:
                edges.add(edge)
            if len(edges) >= edge_limit:
                break
        if len(edges) >= edge_limit:
            break

    if not edges:
        return None

    xs = []
    ys = []
    zs = []
    for u, v in edges:
        xs.extend([vertices[u, 0], vertices[v, 0], None])
        ys.extend([vertices[u, 1], vertices[v, 1], None])
        zs.extend([vertices[u, 2], vertices[v, 2], None])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": "#111827", "width": 2},
        hoverinfo="skip",
        showlegend=False,
        name="wireframe",
    )


def build_plotly_figure(glb_path, settings=None):
    """Build a Plotly mesh figure from a GLB file."""
    settings = {**get_default_viewer_settings(), **(settings or {})}
    loaded = trimesh.load(glb_path, force="scene")
    meshes = loaded.dump(concatenate=False) if isinstance(loaded, trimesh.Scene) else [loaded]

    figure = go.Figure()
    has_geometry = False
    failed_meshes = 0

    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces is None or len(mesh.faces) == 0:
            continue

        try:
            vertices = np.asarray(mesh.vertices, dtype=float)
            faces = np.asarray(mesh.faces, dtype=int)

            if vertices.size == 0 or faces.size == 0:
                failed_meshes += 1
                continue

            faces = _downsample_faces(faces, int(settings["max_faces"]))

            used = np.unique(faces.reshape(-1))
            compact_vertices = vertices[used]
            remap = {int(old): idx for idx, old in enumerate(used)}
            compact_faces = np.array([[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in faces], dtype=int)

            if settings.get("normalize", True):
                compact_vertices = _normalize_vertices(compact_vertices)
        except Exception:
            failed_meshes += 1
            continue

        i = compact_faces[:, 0]
        j = compact_faces[:, 1]
        k = compact_faces[:, 2]
        color = "#b8b8b8"
        vertex_colors = None
        visual = getattr(mesh, "visual", None)
        material = getattr(visual, "material", None) if visual is not None else None
        raw_vertex_colors = getattr(visual, "vertex_colors", None) if visual is not None else None
        if raw_vertex_colors is not None:
            try:
                vertex_array = np.asarray(raw_vertex_colors)
                if vertex_array.ndim == 2 and len(vertex_array) == len(vertices):
                    vertex_colors = _to_plotly_rgba_colors(vertex_array[used])
            except Exception:
                vertex_colors = None
        base_color = getattr(material, "baseColorFactor", None)
        if base_color and len(base_color) >= 3:
            rgb = tuple(int(channel) for channel in base_color[:3])
            color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

        mesh_trace_kwargs = {
            "x": compact_vertices[:, 0],
            "y": compact_vertices[:, 1],
            "z": compact_vertices[:, 2],
            "i": i,
            "j": j,
            "k": k,
            "flatshading": bool(settings["flatshading"]),
            "lighting": {
                "ambient": float(settings["ambient"]),
                "diffuse": float(settings["diffuse"]),
                "specular": float(settings["specular"]),
                "roughness": float(settings["roughness"]),
            },
            "lightposition": {"x": 80, "y": 80, "z": 130},
            "hoverinfo": "skip",
            "name": os.path.basename(glb_path),
            "showscale": False,
        }
        if vertex_colors:
            mesh_trace_kwargs["vertexcolor"] = vertex_colors
        else:
            mesh_trace_kwargs["color"] = color

        figure.add_trace(go.Mesh3d(**mesh_trace_kwargs))

        if settings.get("wireframe"):
            wire = _build_wireframe_trace(
                compact_vertices,
                compact_faces,
                edge_limit=int(settings["wireframe_limit"]),
            )
            if wire is not None:
                figure.add_trace(wire)

        has_geometry = True

    if not has_geometry:
        if failed_meshes:
            raise ValueError("GLB mesh could not be triangulated cleanly")
        raise ValueError("No mesh geometry found in GLB file")

    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor=settings["bg_color"],
        scene={
            "bgcolor": settings["bg_color"],
            "aspectmode": "data",
            "xaxis": {"visible": bool(settings["show_axes"]), "showgrid": False, "zeroline": False},
            "yaxis": {"visible": bool(settings["show_axes"]), "showgrid": False, "zeroline": False},
            "zaxis": {"visible": bool(settings["show_axes"]), "showgrid": False, "zeroline": False},
            "camera": {
                "eye": _camera_eye_from_preset(settings["camera_preset"]),
                "projection": {"type": settings["projection"]},
            },
        },
        showlegend=False,
    )
    return figure


def display_3d_model(glb_path, chart_key=None):
    """Display a 3D GLB model using Plotly as a browser-safe fallback."""
    st.caption(f"Viewing: {os.path.basename(glb_path)}")

    viewer_key = chart_key or "viewer"

    show_preview_mode_badge("Interactive")

    defaults = get_default_viewer_settings()
    settings = {
        "max_faces": st.session_state.get(f"{viewer_key}_max_faces", defaults["max_faces"]),
        "normalize": st.session_state.get(f"{viewer_key}_normalize", defaults["normalize"]),
        "wireframe": st.session_state.get(f"{viewer_key}_wireframe", defaults["wireframe"]),
        "wireframe_limit": st.session_state.get(f"{viewer_key}_wireframe_limit", defaults["wireframe_limit"]),
        "ambient": st.session_state.get(f"{viewer_key}_ambient", defaults["ambient"]),
        "diffuse": st.session_state.get(f"{viewer_key}_diffuse", defaults["diffuse"]),
        "specular": st.session_state.get(f"{viewer_key}_specular", defaults["specular"]),
        "roughness": st.session_state.get(f"{viewer_key}_roughness", defaults["roughness"]),
        "flatshading": st.session_state.get(f"{viewer_key}_flatshading", defaults["flatshading"]),
        "bg_color": st.session_state.get(f"{viewer_key}_bg_color", defaults["bg_color"]),
        "camera_preset": st.session_state.get(f"{viewer_key}_camera_preset", defaults["camera_preset"]),
        "projection": st.session_state.get(f"{viewer_key}_projection", defaults["projection"]),
        "show_axes": st.session_state.get(f"{viewer_key}_show_axes", defaults["show_axes"]),
    }

    with st.expander("Viewer settings", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            settings["max_faces"] = st.slider(
                "Max faces per mesh",
                min_value=5000,
                max_value=220000,
                value=int(settings["max_faces"]),
                step=5000,
                key=f"{viewer_key}_max_faces",
            )
            settings["camera_preset"] = st.selectbox(
                "Camera",
                ["Isometric", "Front", "Top", "Left"],
                index=["Isometric", "Front", "Top", "Left"].index(settings["camera_preset"]),
                key=f"{viewer_key}_camera_preset",
            )
            settings["projection"] = st.selectbox(
                "Projection",
                ["perspective", "orthographic"],
                index=["perspective", "orthographic"].index(settings["projection"]),
                key=f"{viewer_key}_projection",
            )
            settings["bg_color"] = st.color_picker(
                "Background",
                value=settings["bg_color"],
                key=f"{viewer_key}_bg_color",
            )
        with col_b:
            settings["ambient"] = st.slider("Ambient light", 0.0, 1.0, float(settings["ambient"]), 0.05, key=f"{viewer_key}_ambient")
            settings["diffuse"] = st.slider("Diffuse light", 0.0, 1.0, float(settings["diffuse"]), 0.05, key=f"{viewer_key}_diffuse")
            settings["specular"] = st.slider("Specular", 0.0, 1.0, float(settings["specular"]), 0.05, key=f"{viewer_key}_specular")
            settings["roughness"] = st.slider("Roughness", 0.0, 1.0, float(settings["roughness"]), 0.05, key=f"{viewer_key}_roughness")

            settings["flatshading"] = st.checkbox("Flat shading", value=bool(settings["flatshading"]), key=f"{viewer_key}_flatshading")
            settings["normalize"] = st.checkbox("Normalize scale", value=bool(settings["normalize"]), key=f"{viewer_key}_normalize")
            settings["show_axes"] = st.checkbox("Show axes", value=bool(settings["show_axes"]), key=f"{viewer_key}_show_axes")
            settings["wireframe"] = st.checkbox("Wireframe overlay", value=bool(settings["wireframe"]), key=f"{viewer_key}_wireframe")
            if settings["wireframe"]:
                settings["wireframe_limit"] = st.slider(
                    "Wireframe edge limit",
                    min_value=500,
                    max_value=50000,
                    value=int(settings["wireframe_limit"]),
                    step=500,
                    key=f"{viewer_key}_wireframe_limit",
                )

    try:
        figure = build_plotly_figure(glb_path, settings=settings)
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
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
        key=chart_key,
    )


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
        initial_sidebar_state="expanded",
    )

    setup_pwa()
    render_api_key_inputs()

    st.title("StudioRK3dps → 🖼️ → 🧊")

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

    tab_details, tab_prompt, tab_chat, tab1, tab2, tab3, tab4_obj, tab4_stl = st.tabs([
        "Details",
        "3D Image Prompt Ideas",
        "AI Chat Assistant",
        "Image Generation",
        "3D Model Generation",
        "3D File Viewer",
        "Convert to OBJ",
        "Convert to STL",
    ])

    def sanitize_filename(name):
        import re

        return re.sub(r"[^\w\-]+", "_", (name or "").strip())

    with tab_details:
        st.header("Welcome to StudioRK3dps Community App & API Workflow Adaptation")
        st.markdown(
            """
APIs can be inserted in the sidebar on the left — click the arrow to open it.

- **AI Chat Assistant** can give you ideas.
- **Image Generation** can generate an image.
- **3D Model Generation** can generate 3D models.
- **3D File Viewer** is for viewing models.
- **Convert to OBJ** can convert a model's format to OBJ.
- **Convert to STL** can convert a model's format to STL.

---
*Attributions to API tech providers and frameworks.*
"""
        )

    with tab1:
        st.caption("Remember to check API usage!")

        if "image_provider" not in st.session_state:
            st.session_state["image_provider"] = _get_image_provider_name()

        provider_options = _provider_options()
        provider = st.selectbox(
            "Image provider",
            provider_options,
            index=provider_options.index(st.session_state.get("image_provider", "OpenAI")) if st.session_state.get("image_provider", "OpenAI") in provider_options else 0,
            key="image_provider",
        )

        if provider == "OpenAI":
            if "openai_image_model_select" not in st.session_state:
                st.session_state["openai_image_model_select"] = get_openai_image_model_name() or "gpt-image-1"

            image_model_options = _openai_image_model_options()
            if st.session_state["openai_image_model_select"] not in image_model_options:
                image_model_options.insert(0, st.session_state["openai_image_model_select"])

            selected_model = st.selectbox(
                "OpenAI image model",
                image_model_options,
                index=image_model_options.index(st.session_state["openai_image_model_select"]),
                key="openai_image_model_select",
            )
            os.environ["OPENAI_IMAGE_MODEL"] = selected_model
        else:
            st.caption(f"Using {provider} for image generation. Add the matching API key in the sidebar or your .env file.")

        prompt = st.text_area(
            "Image prompt",
            value="One Single Stylized simple multicoloured Common Holly performing Heterophylly whilst presented on a sturdy figurine base suitable for 3D printing.",
        )

        if st.button("Generate Image"):
            with st.spinner(f"Generating image with {provider}..."):
                try:
                    image_path = generate_image_with_provider(prompt, provider)
                    if image_path:
                        st.session_state["image_path"] = image_path
                        st.image(image_path, caption=f"Generated Image ({provider})", width=400)
                except Exception as exc:
                    st.error(str(exc))

    with tab2:
        st.caption("Remember to check API usage!")
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
                st.error("Enter a Cohere API key in the sidebar (or set COHERE_API_KEY).")
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

    with tab_prompt:
        st.subheader("3D Image Prompt Ideas")
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
                st.error("Enter a Cohere API key in the sidebar (or set COHERE_API_KEY).")
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

def get_openai_client():
    from openai import OpenAI

    api_key = get_api_key("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or secrets")
    return OpenAI(api_key=api_key)


def _friendly_openai_image_error(exc):
    """Map known OpenAI image API failures to user-friendly guidance."""
    text = str(exc)
    lowered = text.lower()

    if "billing_hard_limit_reached" in lowered or "billing hard limit has been reached" in lowered:
        return (
            "OpenAI billing limit reached for the active key/project. "
            "Use a funded key, then update OPENAI_API_KEY in your .env or Streamlit secrets and try again."
        )

    if "invalid_api_key" in lowered or "incorrect api key" in lowered:
        return (
            "OpenAI rejected the API key. Update OPENAI_API_KEY in your .env or Streamlit secrets, "
            "then retry image generation."
        )

    if "does not exist" in lowered and "model" in lowered:
        return (
            "The selected OpenAI image model is unavailable for this key/project. "
            "Set OPENAI_IMAGE_MODEL to a supported image model (for example, dall-e-3 or dall-e-2) and retry."
        )
    if "none of the candidate openai image models are available" in lowered:
        return (
            "None of the candidate OpenAI image models are available for this key/project. "
            "Set OPENAI_IMAGE_MODEL to a supported model for your account and retry."
        )

    return f"Image generation failed: {text}"


def _is_expected_openai_image_error(exc):
    lowered = str(exc).lower()
    expected_markers = (
        "billing_hard_limit_reached",
        "billing hard limit has been reached",
        "invalid_api_key",
        "incorrect api key",
        "none of the candidate openai image models are available",
        "openai_api_key not found",
    )
    return all(marker in lowered for marker in ("does not exist", "model")) or any(
        marker in lowered for marker in expected_markers
    )


def get_openai_image_model_name():
    """Return the configured OpenAI image model from UI state, env, or secrets."""
    try:
        ui_model = (st.session_state.get("openai_image_model_select") or "").strip()
        if ui_model:
            return ui_model
    except Exception:
        pass

    load_dotenv(override=True)
    configured = (os.environ.get("OPENAI_IMAGE_MODEL") or "").strip()
    if configured:
        return configured
    try:
        configured = (st.secrets.get("OPENAI_IMAGE_MODEL") or "").strip()
    except Exception:
        configured = ""
    return configured


def _openai_image_model_options():
    """Return supported OpenAI image models as a user-friendly dropdown list."""
    options = ["gpt-image-1", "dall-e-3", "dall-e-2"]
    configured = get_openai_image_model_name()
    if configured and configured not in options:
        options.insert(0, configured)
    return options


def _openai_image_model_candidates():
    configured = get_openai_image_model_name()
    defaults = ["gpt-image-1", "dall-e-3", "dall-e-2"]

    discovered = []
    try:
        client = get_openai_client()
        for model in getattr(client.models, "list", lambda: [])().data:
            model_name = getattr(model, "id", "") or ""
            if not model_name:
                continue
            lowered = model_name.lower()
            if lowered.startswith("gpt-image") or lowered.startswith("dall-e"):
                discovered.append(model_name)
    except Exception:
        discovered = []

    candidates = []
    if configured:
        candidates.append(configured)
    candidates.extend(discovered)
    candidates.extend(defaults)

    seen = set()
    ordered = []
    for model in candidates:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


def _is_unknown_openai_image_model_error(exc):
    lowered = str(exc).lower()
    return "does not exist" in lowered and "model" in lowered


def generate_image_openai(prompt: str) -> str:
    try:
        if not get_api_key("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment or secrets")

        client = get_openai_client()
        response = None
        image_model_error = None
        attempted_models = []
        unknown_model_names = []
        for model_name in _openai_image_model_candidates():
            attempted_models.append(model_name)
            try:
                response = client.images.generate(
                    model=model_name,
                    prompt=prompt,
                    n=1,
                    size="1024x1024",
                )
                break
            except Exception as model_exc:
                if _is_unknown_openai_image_model_error(model_exc):
                    image_model_error = model_exc
                    unknown_model_names.append(model_name)
                    continue
                raise

        if response is None:
            if image_model_error and attempted_models and len(unknown_model_names) == len(attempted_models):
                tried = ", ".join(unknown_model_names)
                raise ValueError(
                    "None of the candidate OpenAI image models are available for this key/project. "
                    f"Tried: {tried}. Set OPENAI_IMAGE_MODEL to a supported model and retry."
                )
            if image_model_error:
                raise image_model_error
            raise RuntimeError("OpenAI image generation failed without a response.")

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
        st.error(_friendly_openai_image_error(e))
        if not _is_expected_openai_image_error(e):
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
                "Authorization": "Bearer " + stability_key,
                "Accept": "model/gltf-binary",
            },
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



if __name__ == "__main__":
    run_app()