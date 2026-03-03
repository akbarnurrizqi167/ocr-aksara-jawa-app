"""
Streamlit App — OCR Aksara Jawa (Javanese Script OCR)

Aplikasi web untuk mengenali aksara Jawa dalam gambar naskah kuno
menggunakan pipeline YOLOv8 + CRNN + CTC Decoding.
"""
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'opencv-python', '-y'], capture_output=True)

import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Add app directory to path
APP_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(APP_DIR))

from pipeline import OCRPipeline

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="OCR Aksara Jawa",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# Custom CSS — Warm Brown Professional Theme
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;500;600;700&display=swap');

    /* ---- Global ---- */
    .stApp {
        font-family: 'Source Sans 3', sans-serif;
        background-color: #faf6f1;
        color: #3d2b1f;
    }

    /* ---- Header ---- */
    .app-header {
        background-color: #3d2b1f;
        border-radius: 0;
        padding: 2.5rem 3rem;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
    }
    .app-header h1 {
        font-family: 'Playfair Display', serif;
        color: #f5e6d3;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .app-header p {
        color: #c4a882;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .header-tag {
        display: inline-block;
        background-color: #5c3d2e;
        color: #d4b896;
        padding: 0.2rem 0.8rem;
        border-radius: 3px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-right: 0.5rem;
        margin-bottom: 0.8rem;
    }

    /* ---- Pipeline Steps (Desktop) ---- */
    .pipeline-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.3rem;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #e0d5c8;
    }
    .pipeline-step {
        background-color: #f0e6d9;
        border: 1px solid #d4c4b0;
        padding: 0.5rem 1rem;
        color: #5c3d2e;
        font-size: 0.78rem;
        font-weight: 600;
        text-align: center;
        white-space: nowrap;
    }
    .pipeline-arrow {
        color: #a0826d;
        font-size: 1rem;
        font-weight: 400;
    }

    /* ---- Pipeline Steps (Mobile) ---- */
    .pipeline-mobile {
        display: none;
        text-align: center;
        padding: 0.8rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e0d5c8;
        color: #5c3d2e;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* ---- Section Headers ---- */
    .section-title {
        font-family: 'Playfair Display', serif;
        color: #3d2b1f;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #c4a882;
        display: inline-block;
    }

    /* ---- Result Box ---- */
    .result-box {
        background-color: #3d2b1f;
        border-left: 4px solid #c4a882;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }
    .result-box .label-text {
        color: #c4a882;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-box .main-text {
        color: #f5e6d3;
        font-size: 1.2rem;
        font-weight: 400;
        line-height: 1.9;
        word-spacing: 2px;
    }

    /* ---- Info Panel ---- */
    .info-panel {
        background-color: #f0e6d9;
        border: 1px solid #d4c4b0;
        padding: 1.2rem 1.5rem;
    }
    .info-panel p {
        color: #5c3d2e;
        font-size: 0.85rem;
        margin: 0.2rem 0;
    }
    .info-panel .info-label {
        color: #8b6914;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ---- Metric Row ---- */
    .metric-row {
        display: flex;
        gap: 0;
        margin: 1rem 0;
    }
    .metric-item {
        flex: 1;
        background-color: #f0e6d9;
        border: 1px solid #d4c4b0;
        border-right: none;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-item:last-child {
        border-right: 1px solid #d4c4b0;
    }
    .metric-item .m-label {
        color: #8b6914;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }
    .metric-item .m-value {
        color: #3d2b1f;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .metric-item .m-unit {
        color: #8b6914;
        font-size: 0.7rem;
        font-weight: 400;
    }

    /* ---- Upload Area ---- */
    .upload-section {
        background-color: #f0e6d9;
        border: 1px dashed #b89d7a;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }

    /* ---- Controls ---- */
    .controls-row {
        background-color: #f0e6d9;
        border: 1px solid #d4c4b0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    /* ---- Empty State ---- */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #8b6914;
    }
    .empty-state h3 {
        font-family: 'Playfair Display', serif;
        color: #5c3d2e;
        font-weight: 600;
        font-size: 1.3rem;
    }
    .empty-state p {
        color: #8b6914;
        font-size: 0.9rem;
        max-width: 450px;
        margin: 0.5rem auto;
    }

    /* ---- Hide Streamlit defaults ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    section[data-testid="stSidebar"] {display: none;}

    /* ---- Streamlit overrides for brown theme ---- */
    .stButton > button[kind="primary"] {
        background-color: #3d2b1f;
        color: #f5e6d3;
        border: none;
        border-radius: 0;
        font-weight: 600;
        letter-spacing: 0.5px;
        padding: 0.6rem 2rem;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #5c3d2e;
        color: #f5e6d3;
    }

    /* Slider overrides */
    .stSlider label {
        color: #3d2b1f !important;
        font-weight: 600 !important;
    }

    /* Dataframe text */
    .stDataFrame {
        border: 1px solid #d4c4b0;
    }

    /* ============================================ */
    /* Responsive — Tablet (max 768px)              */
    /* ============================================ */
    @media (max-width: 768px) {
        .app-header {
            padding: 1.5rem 1.2rem !important;
        }
        .app-header h1 {
            font-size: 1.6rem !important;
        }
        .app-header p {
            font-size: 0.85rem !important;
        }

        .pipeline-bar {
            display: none !important;
        }
        .pipeline-mobile {
            display: block !important;
        }

        .metric-row {
            flex-wrap: wrap !important;
        }
        .metric-item {
            flex: 1 1 45% !important;
            border-right: 1px solid #d4c4b0 !important;
        }

        .result-box {
            padding: 1rem 1.2rem !important;
        }
        .result-box .main-text {
            font-size: 1rem !important;
        }

        .section-title {
            font-size: 1.1rem !important;
        }
    }

    /* ============================================ */
    /* Responsive — Mobile (max 480px)              */
    /* ============================================ */
    @media (max-width: 480px) {
        .app-header {
            padding: 1.2rem 1rem !important;
            margin: -1rem -1rem 1rem -1rem !important;
        }
        .app-header h1 {
            font-size: 1.3rem !important;
        }
        .app-header p {
            font-size: 0.75rem !important;
        }
        .header-tag {
            font-size: 0.6rem !important;
            padding: 0.15rem 0.5rem !important;
        }

        .metric-row {
            flex-direction: column !important;
        }
        .metric-item {
            border-right: 1px solid #d4c4b0 !important;
            border-bottom: none !important;
        }
        .metric-item:last-child {
            border-bottom: 1px solid #d4c4b0 !important;
        }
        .metric-item .m-value {
            font-size: 1.2rem !important;
        }

        .result-box {
            padding: 0.8rem 1rem !important;
        }
        .result-box .main-text {
            font-size: 0.9rem !important;
            line-height: 1.6 !important;
        }

        .info-panel {
            padding: 0.8rem 1rem !important;
        }

        .empty-state {
            padding: 2rem 1rem !important;
        }
        .empty-state h3 {
            font-size: 1.1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Model Loading (Cached)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load OCR pipeline models (cached across sessions)"""
    models_dir = APP_DIR / "models"

    pipeline = OCRPipeline(
        yolo_model_path=str(models_dir / "yolov8_best.pt"),
        crnn_model_path=str(models_dir / "crnn_best.pt"),
        charset_path=str(models_dir / "charset.txt"),
        device='cpu',
        yolo_conf=0.25,
        crnn_img_size=(32, 128)
    )

    return pipeline


def draw_annotated_image(image: np.ndarray, results: dict) -> np.ndarray:
    """Draw bounding boxes and labels on image using PIL for Unicode support"""
    annotated = image.copy()

    # Convert to PIL for text rendering (supports Unicode e, e)
    pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Try to load a font that supports Unicode, fallback to default
    font_size = 16
    try:
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSText.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, font_size)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    for i, r in enumerate(results['results']):
        x1, y1, x2, y2 = r['bbox']

        conf = r['recognition_conf']
        # Warm palette: brown tones
        if conf >= 0.8:
            color = (90, 140, 70)     # Muted green
        elif conf >= 0.5:
            color = (180, 140, 60)    # Warm amber
        else:
            color = (180, 80, 60)     # Muted red

        label = f"{i+1}: {r['text']} ({conf:.0%})"

        bbox_text = draw.textbbox((0, 0), label, font=font)
        label_w = bbox_text[2] - bbox_text[0]
        label_h = bbox_text[3] - bbox_text[1]

        label_bg_y1 = max(0, y1 - label_h - 8)
        draw.rectangle(
            [(x1, label_bg_y1), (x1 + label_w + 8, y1)],
            fill=color
        )
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        draw.text((x1 + 4, label_bg_y1 + 2), label, fill=(255, 255, 255), font=font)

    annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return annotated


# ============================================================
# Main Content
# ============================================================

# Header
st.markdown("""
<div class="app-header">
    <div>
        <span class="header-tag">Deep Learning</span>
        <span class="header-tag">YOLOv8 + CRNN</span>
    </div>
    <h1>OCR Aksara Jawa</h1>
    <p>Optical Character Recognition untuk naskah kuno aksara Jawa</p>
</div>
""", unsafe_allow_html=True)

# Pipeline visualization — desktop (hidden on tablet/mobile via CSS)
st.markdown("""
<div class="pipeline-bar">
    <div class="pipeline-step">Input Image</div>
    <span class="pipeline-arrow">&rarr;</span>
    <div class="pipeline-step">YOLOv8 Detection</div>
    <span class="pipeline-arrow">&rarr;</span>
    <div class="pipeline-step">Crop Region</div>
    <span class="pipeline-arrow">&rarr;</span>
    <div class="pipeline-step">CRNN Recognition</div>
    <span class="pipeline-arrow">&rarr;</span>
    <div class="pipeline-step">CTC Decode</div>
</div>
<div class="pipeline-mobile">
    Input Image &rarr; YOLOv8 &rarr; Crop &rarr; CRNN &rarr; CTC Decode
</div>
""", unsafe_allow_html=True)

# Controls row
col_upload, col_conf = st.columns([3, 1])

with col_conf:
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Minimum confidence untuk deteksi kata oleh YOLOv8"
    )

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload gambar naskah aksara Jawa",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload gambar naskah kuno yang berisi aksara Jawa untuk dikenali"
    )


if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Gagal membaca gambar. Pastikan file adalah gambar yang valid.")
    else:
        # Display original image
        col_orig, col_info = st.columns([3, 1])
        with col_orig:
            st.markdown('<div class="section-title">Gambar Input</div>', unsafe_allow_html=True)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, use_container_width=True)

        with col_info:
            h, w = image.shape[:2]
            st.markdown(f"""
            <div class="info-panel">
                <div class="info-label">Info Gambar</div>
                <p><strong>File:</strong> {uploaded_file.name}</p>
                <p><strong>Ukuran:</strong> {w} x {h} px</p>
                <p><strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Process button
        if st.button("Proses OCR", type="primary", use_container_width=True):

            # Load pipeline
            with st.spinner("Memuat model..."):
                pipeline = load_pipeline()
                pipeline.yolo_conf = conf_threshold

            # Run pipeline
            with st.spinner("Memproses gambar..."):
                results = pipeline.process_image(image)

            # ========================================
            # Results Display
            # ========================================

            if results['num_detections'] == 0:
                st.warning("Tidak ada kata yang terdeteksi. Coba turunkan confidence threshold.")
            else:
                # Timing Metrics
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-item">
                        <div class="m-label">Kata Terdeteksi</div>
                        <div class="m-value">{results['num_detections']}</div>
                        <div class="m-unit">words</div>
                    </div>
                    <div class="metric-item">
                        <div class="m-label">Deteksi</div>
                        <div class="m-value">{results['timing']['detection_ms']:.0f}</div>
                        <div class="m-unit">ms</div>
                    </div>
                    <div class="metric-item">
                        <div class="m-label">Pengenalan</div>
                        <div class="m-value">{results['timing']['recognition_ms']:.0f}</div>
                        <div class="m-unit">ms</div>
                    </div>
                    <div class="metric-item">
                        <div class="m-label">Total</div>
                        <div class="m-value">{results['timing']['total_ms']:.0f}</div>
                        <div class="m-unit">ms</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Full Text Output
                st.markdown(f"""
                <div class="result-box">
                    <div class="label-text">Hasil Transliterasi Lengkap</div>
                    <div class="main-text">{results['full_text']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")

                # Detection + Table
                col_det, col_table = st.columns([3, 2])

                with col_det:
                    st.markdown('<div class="section-title">Hasil Deteksi</div>', unsafe_allow_html=True)
                    annotated_img = draw_annotated_image(image, results)
                    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True)

                with col_table:
                    st.markdown('<div class="section-title">Detail per Kata</div>', unsafe_allow_html=True)

                    table_data = []
                    for i, r in enumerate(results['results']):
                        table_data.append({
                            "No": i + 1,
                            "Kata": r['text'],
                            "Deteksi": f"{r['detection_conf']:.0%}",
                            "Pengenalan": f"{r['recognition_conf']:.0%}",
                        })

                    df = pd.DataFrame(table_data)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        height=min(400, 35 * len(table_data) + 55)
                    )

                # Divider
                st.markdown(
                    '<hr style="border: none; border-top: 1px solid #d4c4b0; margin: 2rem 0;">',
                    unsafe_allow_html=True
                )

                # Cropped Words Grid
                st.markdown('<div class="section-title">Kata Terdeteksi (Cropped)</div>', unsafe_allow_html=True)

                num_cols = 5
                cols = st.columns(num_cols)

                for i, r in enumerate(results['results']):
                    crop = r.get('crop')
                    if crop is not None and crop.size > 0:
                        with cols[i % num_cols]:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            st.image(crop_rgb, use_container_width=True)
                            st.caption(f"**{r['text']}** — {r['recognition_conf']:.0%}")

                # Divider
                st.markdown(
                    '<hr style="border: none; border-top: 1px solid #d4c4b0; margin: 2rem 0;">',
                    unsafe_allow_html=True
                )

                # Raw Output (Expandable)
                with st.expander("Detail Teknis (Raw CTC Output)"):
                    for i, r in enumerate(results['results']):
                        raw = r.get('raw_output', 'N/A')
                        st.markdown(
                            f"**Kata {i+1}:** `{r['text']}` "
                            f"← Raw: `{raw}` "
                            f"| Det: {r['detection_conf']:.3f} "
                            f"| Rec: {r['recognition_conf']:.3f}"
                        )

else:
    # Empty state
    st.markdown("""
    <div class="empty-state">
        <h3>Upload Gambar untuk Memulai</h3>
        <p>
            Upload gambar naskah kuno yang berisi aksara Jawa.
            Aplikasi akan mendeteksi kata-kata dan mengenali karakternya
            secara otomatis menggunakan AI.
        </p>
    </div>
    """, unsafe_allow_html=True)
