"""Interactive Streamlit dashboard for the SRM FastAPI service."""

from __future__ import annotations

import json
import queue
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pydeck as pdk
import rasterio
import streamlit as st
import torch
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.core import inference as inference_service

ProgressResult = TypeVar("ProgressResult")


@st.cache_resource
def _get_local_model() -> tuple[torch.nn.Module, torch.device]:
    """Load the existing checkpoint once per Streamlit process."""
    return inference_service.MODEL, inference_service.DEVICE


def _run_local_inference(
    raster_bytes: bytes,
    progress_callback: Callable[[int, int], None],
) -> dict[str, object]:
    """Run the API pipeline locally for Streamlit-only deployment."""
    _, device = _get_local_model()
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as uploaded:
        uploaded.write(raster_bytes)
        input_path = Path(uploaded.name)

    output_path: Path | None = None
    geojson_path: Path | None = None
    try:
        with rasterio.open(input_path) as source:
            if source.count < 3:
                raise ValueError(f"Input raster must contain at least 3 bands; received {source.count}")
            width, height = source.width, source.height
            read_width, read_height = width, height
            if max(width, height) > 2048:
                factor = 2048 / max(width, height)
                read_width, read_height = max(1, round(width * factor)), max(1, round(height * factor))
                rgb_data = source.read(
                    [1, 2, 3],
                    out_shape=(3, read_height, read_width),
                    resampling=Resampling.bilinear,
                    masked=True,
                )
            else:
                rgb_data = source.read([1, 2, 3], masked=True)
            source_data = inference_service._percentile_normalize_raster(rgb_data)
            source_profile = source.profile.copy()
            source_transform = source.transform
            source_crs = source_profile.get("crs")
            source_bounds = source.bounds

        input_tensor = torch.from_numpy(source_data).unsqueeze(0).to(device)
        with torch.no_grad():
            baseline_array, model_array = inference_service._infer_in_tiles(
                input_tensor,
                progress_callback=progress_callback,
            )
        del input_tensor
        inference_service._clear_torch_memory()

        baseline_array = np.clip(baseline_array, 0.0, 1.0)
        model_array = np.clip(model_array, 0.0, 1.0)
        matched_array = inference_service._match_color_statistics(model_array, baseline_array)
        baseline_uint8 = np.clip(baseline_array * 255.0, 0, 255).astype(np.uint8)
        model_uint8 = np.clip(matched_array * 255.0, 0, 255).astype(np.uint8)
        psnr = float(peak_signal_noise_ratio(baseline_uint8, model_uint8, data_range=255))
        ssim = float(
            structural_similarity(
                baseline_uint8, model_uint8, channel_axis=-1, data_range=255
            )
        )

        with tempfile.NamedTemporaryFile(suffix="_srm_super_resolved.tif", delete=False) as output:
            output_path = Path(output.name)
        output_profile = source_profile.copy()
        output_profile.update(
            driver="GTiff",
            width=model_uint8.shape[1],
            height=model_uint8.shape[0],
            count=model_uint8.shape[2],
            dtype="uint8",
            transform=source_transform * Affine.scale(
                width / model_uint8.shape[1], height / model_uint8.shape[0]
            ),
            compress="deflate",
            nodata=None,
        )
        with rasterio.open(output_path, "w", **output_profile) as destination:
            destination.write(np.moveaxis(model_uint8, -1, 0))

        with tempfile.NamedTemporaryFile(suffix="_srm_vector_mapping.geojson", delete=False) as vector:
            geojson_path = Path(vector.name)
        feature_count = inference_service._extract_vector_mapping(
            model_uint8,
            output_profile["transform"],
            source_crs,
            str(geojson_path),
        )
        return {
            "output_path": str(output_path),
            "geojson_path": str(geojson_path),
            "geojson_feature_count": feature_count,
            "scale_factor": 4,
            "width": model_uint8.shape[1],
            "height": model_uint8.shape[0],
            "bounds": {
                "left": source_bounds.left,
                "bottom": source_bounds.bottom,
                "right": source_bounds.right,
                "top": source_bounds.top,
            },
            "crs": source_crs.to_string() if source_crs is not None else None,
            "psnr_db": psnr,
            "ssim": ssim,
        }
    finally:
        input_path.unlink(missing_ok=True)
        if output_path is None:
            if geojson_path is not None:
                geojson_path.unlink(missing_ok=True)
        elif geojson_path is None:
            output_path.unlink(missing_ok=True)


def _run_with_progress(
    operation: Callable[[], ProgressResult],
    start: int,
    end: int,
    status: str,
    progress_bar: Any,
    status_text: Any,
) -> ProgressResult:
    """Run a blocking operation while smoothly polling progress every 100ms."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="satfeed-stage") as executor:
        future = executor.submit(operation)
        current = float(start)
        while not future.done():
            status_text.markdown(f"**{status}** ({round(current)}%)")
            progress_bar.progress(min(round(current), end - 1))
            current = min(current + max((end - start) / 20, 0.5), end - 1)
            time.sleep(0.1)
        result = future.result()
        status_text.markdown(f"**{status}** ({end}%)")
        progress_bar.progress(end)
        return result


def _raster_preview(raster_bytes: bytes) -> np.ndarray:
    """Build an RGB-compatible preview while preserving RGB channel relationships."""
    with MemoryFile(raster_bytes) as memory_file:
        with memory_file.open() as dataset:
            bands = dataset.read()

    selected = bands[:3]
    if selected.shape[0] == 1:
        selected = np.repeat(selected, 3, axis=0)
    elif selected.shape[0] == 2:
        selected = np.concatenate((selected, selected[:1]), axis=0)

    source_dtype = selected.dtype
    selected = selected.astype(np.float32)
    finite = np.isfinite(selected)
    if source_dtype == np.uint8:
        preview = selected / 255.0
    elif np.any(finite):
        # Match the backend's per-channel 2-98% normalization.
        preview = np.zeros_like(selected)
        for band_index in range(3):
            band = selected[:, :, band_index]
            valid = finite[:, :, band_index]
            if not np.any(valid):
                continue
            minimum, maximum = np.percentile(band[valid], (2.0, 98.0))
            value_range = maximum - minimum
            if value_range > 0:
                preview[:, :, band_index] = np.clip(
                    (band - minimum) / value_range, 0, 1
                )
    else:
        preview = np.zeros_like(selected)
    preview[~finite] = 0.0

    return np.moveaxis(preview, 0, -1)


st.set_page_config(
    page_title="SatFeed",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
    .stApp { background: #0B0F17; color: #F8FAFC; font-family: 'Inter', sans-serif; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { font-family: 'Plus Jakarta Sans', sans-serif; }
    [data-testid="stSidebar"] { background: #101521; border-right: 1px solid rgba(255,255,255,.08); }
    [data-testid="stMetric"] {
        background: rgba(22, 27, 38, .7); border: 1px solid rgba(255,255,255,.08);
        border-radius: 16px; padding: 20px 22px; min-height: 112px;
        box-shadow: 0 14px 32px rgba(0,0,0,.18);
    }
    [data-testid="stMetricLabel"] { color: #A8B0C0; font-size: 0.9rem; }
    [data-testid="stMetricValue"] { color: #F8FAFC; }
    [data-testid="stButton"] > button, [data-testid="stDownloadButton"] > button {
        border-radius: 12px; border: 1px solid rgba(139,92,246,.6);
        transition: all .2s ease; font-family: 'Inter', sans-serif; font-weight: 600;
    }
    [data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        color: #FFFFFF; font-weight: 700; box-shadow: 0 8px 22px rgba(99,102,241,.28);
    }
    [data-testid="stButton"] > button:hover, [data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(135deg, #818CF8, #A78BFA);
        border-color: #A78BFA; transform: translateY(-1px);
    }
    .satfeed-panel {
        background: rgba(22, 27, 38, .7); border: 1px solid rgba(255,255,255,.08);
        border-radius: 16px; padding: 22px 24px; margin: 22px 0;
        box-shadow: 0 18px 40px rgba(0,0,0,.16); backdrop-filter: blur(14px);
    }
    .satfeed-hero {
        display: flex; align-items: center; justify-content: space-between;
        min-height: 210px; margin: 0 0 28px; padding: 30px 38px;
        overflow: hidden; border: 1px solid rgba(255,255,255,.08); border-radius: 20px;
        background: linear-gradient(115deg, #0B0F17 0%, #141A2A 62%, #211B3D 100%);
        box-shadow: 0 20px 50px rgba(0,0,0,.25);
    }
    .satfeed-hero-copy { max-width: 68%; position: relative; z-index: 1; }
    .satfeed-kicker { color: #A78BFA; font-size: .78rem; font-weight: 700;
        letter-spacing: .16em; text-transform: uppercase; margin-bottom: 12px; }
    .satfeed-hero h1 { color: #F8FAFC; font-size: clamp(1.8rem, 4vw, 3.2rem);
        line-height: 1.08; margin: 0 0 12px; letter-spacing: -.04em; }
    .satfeed-hero p { color: #A8B0C0; font-size: 1rem; line-height: 1.6; margin: 0; }
    .satfeed-satellite { width: 190px; height: 150px; opacity: .9; flex: 0 0 auto; }
    .comparison-caption {
        color: #A8B0C0; font-size: 0.9rem; font-weight: 600;
        white-space: nowrap; margin: 0 0 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="satfeed-hero">
      <div class="satfeed-hero-copy">
        <div class="satfeed-kicker">Satellite intelligence platform</div>
        <h1>SatFeed</h1>
        <p>Satellite Imagery Super-Resolution &amp; GIS Analytics</p>
      </div>
      <svg class="satfeed-satellite" viewBox="0 0 220 170" fill="none"
           xmlns="http://www.w3.org/2000/svg" aria-label="Satellite illustration">
        <g stroke="#A78BFA" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
          <path d="M72 82L38 52 18 72l35 31M148 82l34-30 20 20-35 31"/>
          <path d="M72 82h76v42H72zM84 82V61h52v21M82 124l-15 25m71-25 15 25"/>
          <path d="M97 103h26M110 61V39M101 39h18"/>
          <circle cx="110" cy="29" r="7" fill="#6366F1"/>
          <path d="M54 103l-22 22m134-22 22 22"/>
        </g>
        <path d="M12 151c48-18 138-18 196 0" stroke="#6366F1" stroke-width="2" stroke-dasharray="5 8"/>
      </svg>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("SatFeed")
    st.caption("Satellite Imagery Super-Resolution & GIS Analytics")
    st.markdown(
        "Upload a multispectral GeoTIFF and run tiled PyTorch inference directly "
        "inside this Streamlit app."
    )
    st.divider()
    st.subheader("Runtime")
    st.info("Inference runs locally in the Streamlit process.")

st.markdown('<div class="satfeed-panel">', unsafe_allow_html=True)
st.subheader("Upload")

uploaded_file = st.file_uploader(
    "Upload medium-resolution multispectral GeoTIFF",
    type=["tif", "tiff"],
    help="The backend expects a four-band GeoTIFF (for example, RGB + NIR).",
)

demo_path = Path(__file__).resolve().parents[2] / "sample_lr.tif"
if st.button(
    "Load Austin demo (fast presentation example)",
    use_container_width=True,
    disabled=not demo_path.is_file(),
):
    st.session_state["demo_raster_bytes"] = demo_path.read_bytes()
    st.session_state["demo_raster_name"] = "austin_demo.tif"

if uploaded_file is not None:
    st.success(f"Ready: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
    st.caption("The input will be normalized before model inference.")
elif "demo_raster_bytes" in st.session_state:
    st.success("Ready: **Austin demo** (included sample raster)")
    st.caption("The small demo is optimized for a quick presentation run.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="satfeed-panel">', unsafe_allow_html=True)
st.subheader("Processing")
has_demo = "demo_raster_bytes" in st.session_state
run_inference = st.button(
    "Run SatFeed Super-Resolution",
    type="primary",
    use_container_width=True,
    disabled=uploaded_file is None and not has_demo,
)
st.markdown("</div>", unsafe_allow_html=True)

if run_inference and (uploaded_file is not None or has_demo):
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        payload = _run_with_progress(
            uploaded_file.getvalue
            if uploaded_file is not None
            else lambda: st.session_state["demo_raster_bytes"],
            0,
            15,
            "Stage 1/5: Loading GeoTIFF & Extracting RGB Bands...",
            progress_bar,
            status_text,
        )
        _run_with_progress(
            lambda: None,
            15,
            30,
            "Stage 2/5: Normalizing Percentiles & Dynamic Range...",
            progress_bar,
            status_text,
        )
        with st.spinner("Running super-resolution and sub-pixel classification..."):
            tile_updates: queue.Queue[tuple[int, int]] = queue.Queue()
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="satfeed-local-inference") as executor:
                future = executor.submit(
                    _run_local_inference,
                    payload,
                    lambda completed, total: tile_updates.put((completed, total)),
                )
                completed_tiles, total_tiles = 0, 1
                while not future.done() or not tile_updates.empty():
                    try:
                        completed_tiles, total_tiles = tile_updates.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    pct = 30 + int((completed_tiles / total_tiles) * 45)
                    status_text.markdown(
                        f"**Stage 3/5**: Running 4x Super-Resolution on Tile "
                        f"{completed_tiles}/{total_tiles}... ({pct}%)"
                    )
                    progress_bar.progress(pct)
                result = future.result()
                pct = 75
                status_text.markdown(
                    f"**Stage 3/5**: Running 4x Super-Resolution on Tile "
                    f"{total_tiles}/{total_tiles}... ({pct}%)"
                )
                progress_bar.progress(pct)
        progress_bar.progress(75)
        status_text.markdown("**Stage 4/5**: Matching Spectral Colors & Computing PSNR/SSIM...")
        status_text.markdown("**Stage 5/5: Extracting Road & Building Vector Footprints... (90%)**")
        progress_bar.progress(90)
    except (OSError, RuntimeError, ValueError) as exc:
        status_text.empty()
        progress_bar.empty()
        st.error(f"Super-resolution failed: {exc}")
    else:
        try:
            tif_path = Path(result["output_path"])
            geojson_path = Path(result["geojson_path"])
            geotiff_bytes = tif_path.read_bytes()
            geojson_bytes = geojson_path.read_bytes()
        except (KeyError, OSError, TypeError) as exc:
            status_text.empty()
            progress_bar.empty()
            st.error(f"Backend returned incomplete output metadata: {exc}")
            st.stop()
        st.session_state["srm_original"] = payload
        st.session_state["srm_result"] = geotiff_bytes
        st.session_state["srm_result_name"] = "srm_super_resolved.tif"
        st.session_state["srm_geojson"] = geojson_bytes
        st.session_state["srm_geojson_name"] = "srm_vector_mapping.geojson"
        st.session_state["srm_metrics"] = result
        status_text.markdown("**Stage 5/5: Extracting Road & Building Vector Footprints... (100%)**")
        progress_bar.progress(100)
        st.success("Super-resolution and SRM classification completed successfully.")

if "srm_result" in st.session_state:
    metrics = st.session_state["srm_metrics"]
    st.markdown('<div class="satfeed-panel">', unsafe_allow_html=True)
    st.subheader("Metrics")
    metric_columns = st.columns(2)
    metric_columns[0].metric("PSNR", f"{metrics['psnr_db']:.2f} dB")
    metric_columns[1].metric("SSIM", f"{metrics['ssim']:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Image comparison")
    try:
        original_preview = _raster_preview(st.session_state["srm_original"])
        result_preview = _raster_preview(st.session_state["srm_result"])
    except (OSError, ValueError, rasterio.errors.RasterioIOError) as exc:
        st.error(f"Unable to preview the raster outputs: {exc}")
    else:
        preview_columns = st.columns(2)
        with preview_columns[0]:
            st.markdown('<p class="comparison-caption">Original Low-Resolution (2m)</p>', unsafe_allow_html=True)
            st.image(
                original_preview,
                use_container_width=True,
            )
        with preview_columns[1]:
            st.markdown('<p class="comparison-caption">Super-Resolved High-Resolution (0.5m)</p>', unsafe_allow_html=True)
            st.image(
                result_preview,
                use_container_width=True,
            )

    st.divider()
    st.subheader("Map overlay")
    try:
        geojson = json.loads(st.session_state["srm_geojson"])
        with MemoryFile(st.session_state["srm_result"]) as memory_file:
            with memory_file.open() as raster:
                bounds = raster.bounds
                if raster.crs and not raster.crs.is_geographic:
                    bounds = transform_bounds(raster.crs, "EPSG:4326", *bounds)
        left, bottom, right, top = bounds
        center_lat = (bottom + top) / 2
        center_lon = (left + right) / 2
        layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson,
            pickable=True,
            stroked=True,
            filled=False,
            get_line_color=[255, 80, 20, 220],
            get_line_width=3,
        )
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=14,
            pitch=0,
        )
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{feature_type}"},
                map_style=None,
            ),
            use_container_width=True,
        )
    except (json.JSONDecodeError, KeyError, OSError, ValueError, rasterio.errors.RasterioIOError) as exc:
        st.error(f"Unable to render vector overlay: {exc}")

    st.divider()
    st.subheader("Downloads")
    download_columns = st.columns(2)
    with download_columns[0]:
        st.download_button(
            "Download high-resolution GeoTIFF",
            data=st.session_state["srm_result"],
            file_name=st.session_state["srm_result_name"],
            mime="image/tiff",
            type="primary",
            use_container_width=True,
        )
    with download_columns[1]:
        st.download_button(
            "Download building & road vectors",
            data=st.session_state["srm_geojson"],
            file_name=st.session_state["srm_geojson_name"],
            mime="application/geo+json",
            use_container_width=True,
        )
