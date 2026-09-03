import os
import sys
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    CAR_DAMAGE_MODEL_PATH,
    FIRE_SEG_MODEL_PATH,
    DEFAULT_DAMAGE_CONF,
    DEFAULT_FIRE_CONF,
    IMAGES_DIR,
    VIDEOS_DIR
)
from src.damage_detector import CarDamageDetector
from src.fire_segmenter import FireSegmenter

st.set_page_config(
    page_title="CarVision AI - Inspection Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #00e5ff;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #00838f;
        color: white;
        font-weight: bold;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_damage_detector():
    return CarDamageDetector()

@st.cache_resource
def get_fire_segmenter():
    return FireSegmenter()

st.title("🚗 CarVision AI: Automated Damage & Fire Inspection")
st.caption("Fine-Tuned YOLO11m Vehicle Body Defect Recognition & YOLO11n Fire Segmentation")

# Sidebar
st.sidebar.header("⚙️ Inspection Settings")
task_mode = st.sidebar.selectbox(
    "Inspection Task",
    ["Vehicle Damage Detection (YOLO11m)", "Fire & Hazard Segmentation (YOLO11n)"]
)

if "Damage" in task_mode:
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.10, 0.95, float(DEFAULT_DAMAGE_CONF), 0.05)
    polygon_mode = st.sidebar.checkbox("Neon Polygon Contours", value=True)
else:
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.10, 0.95, float(DEFAULT_FIRE_CONF), 0.05)
    polygon_mode = False

st.sidebar.markdown("---")
st.sidebar.info("Model Status: **Online & Ready**\n\n• Car Damage: YOLO11m (6 Classes)\n• Fire Hazard: YOLO11n-seg")

# Input Section
input_type = st.radio("Select Input Source:", ["Upload File", "Choose Sample Media"], horizontal=True)

image_input = None
video_path = None

if input_type == "Upload File":
    uploaded = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "webp", "mp4"])
    if uploaded is not None:
        if uploaded.type.startswith("image"):
            image_input = Image.open(uploaded).convert("RGB")
        elif uploaded.type.startswith("video"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            video_path = tfile.name
else:
    sample_options = []
    if os.path.exists(IMAGES_DIR):
        sample_options += [os.path.join("data/images", f) for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if os.path.exists(VIDEOS_DIR):
        sample_options += [os.path.join("data/videos", f) for f in os.listdir(VIDEOS_DIR) if f.endswith('.mp4')]

    chosen_sample = st.selectbox("Select Sample Asset:", sample_options)
    if chosen_sample:
        if chosen_sample.endswith('.mp4'):
            video_path = chosen_sample
        else:
            image_input = Image.open(chosen_sample).convert("RGB")

# Image Processing
if image_input is not None:
    img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Input")
        st.image(image_input, use_column_width=True)

    with col2:
        st.subheader("🔍 AI Inspection Output")
        with st.spinner("Analyzing image..."):
            if "Damage" in task_mode:
                detector = get_damage_detector()
                annotated_bgr, detections, latency = detector.process_frame(
                    img_bgr, conf=conf_thresh, polygon_mode=polygon_mode
                )
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_column_width=True)

                st.markdown(f"**Latency:** `{latency:.1f} ms` | **Defects Detected:** `{len(detections)}`")

                if detections:
                    st.markdown("### Detected Defect Breakdown")
                    tally = {}
                    for d in detections:
                        cname = d['class_name'].replace('_', ' ').title()
                        tally[cname] = tally.get(cname, 0) + 1

                    cols = st.columns(len(tally))
                    for idx, (defect, count) in enumerate(tally.items()):
                        with cols[idx]:
                            st.metric(defect, f"{count} found")
                else:
                    st.success("✅ No vehicle damage detected above confidence threshold!")

            else:
                segmenter = get_fire_segmenter()
                annotated_bgr, count, latency, _ = segmenter.process_frame(
                    img_bgr, conf=conf_thresh
                )
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_column_width=True)

                st.markdown(f"**Latency:** `{latency:.1f} ms` | **Fire Instances:** `{count}`")
                if count > 0:
                    st.error(f"⚠️ DANGER: {count} Fire hazard instance(s) identified!")
                else:
                    st.success("✅ No fire or smoke hazard detected.")

# Video Processing (Sample snippet)
elif video_path is not None:
    st.subheader("🎥 Video Inspection")
    st.video(video_path)
    if st.button("Run Full Video Inspection Annotation"):
        st.info("Batch video processing started. See outputs/videos/ for the full rendered result.")
        detector = get_damage_detector()
        from src.video_pipeline import VideoPipeline
        pipeline = VideoPipeline(detector)
        out_file = pipeline.annotate_video_file(video_path, conf=conf_thresh, polygon_mode=polygon_mode)
        st.success(f"Video annotation complete! Saved to: `{out_file}`")
        if os.path.exists(out_file):
            st.video(out_file)
