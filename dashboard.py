import os
import io
import cv2
import numpy as np
import streamlit as st
import folium
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from severity_estimator import estimate_pothole_metrics, calculate_road_health_score

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Municipal Road Damage Inspection & Mapping",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚧"
)

# ─────────────────────────────────────────────
# CSS Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 50%, #1a0a2e 100%);
        color: #e2e8f0;
    }
    h1, h2, h3 { color: #f8fafc !important; }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 22px 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        backdrop-filter: blur(8px);
        margin-bottom: 8px;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-label { color: #94a3b8; font-size: 12px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; }
    .metric-val { font-size: 36px; font-weight: 800; margin-top: 6px; line-height: 1.1; }
    .metric-sub { font-size: 12px; color: #64748b; margin-top: 4px; }

    /* Priority Badge */
    .badge-critical { background:#ef4444; color:#fff; padding:4px 12px; border-radius:999px; font-weight:700; font-size:13px; }
    .badge-high    { background:#f97316; color:#fff; padding:4px 12px; border-radius:999px; font-weight:700; font-size:13px; }
    .badge-medium  { background:#eab308; color:#000; padding:4px 12px; border-radius:999px; font-weight:700; font-size:13px; }
    .badge-low     { background:#22c55e; color:#fff; padding:4px 12px; border-radius:999px; font-weight:700; font-size:13px; }

    /* Section divider */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
        margin: 24px 0;
    }

    /* Mode tabs */
    .stRadio [role=radiogroup] { flex-direction: row; gap: 12px; }
    .stRadio [role=radiogroup] label {
        background: rgba(30,41,59,0.7);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 10px;
        padding: 10px 18px;
        cursor: pointer;
        color: #e2e8f0;
    }

    /* Upload area */
    .uploadedFile { border-radius: 12px; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
        border-right: 1px solid rgba(99,102,241,0.2);
    }

    /* Health bar */
    .health-bar-container {
        background: rgba(15,23,42,0.8);
        border-radius: 999px;
        height: 12px;
        width: 100%;
        margin-top: 8px;
        overflow: hidden;
    }
    .health-bar-fill {
        height: 12px;
        border-radius: 999px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load YOLO Model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_yolo_model():
    model_path = r"c:\Users\meher\Downloads\mini DATASET\models\best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    st.warning("⚠️ Trained model not found at models/best.pt — using default YOLOv8-Nano weights.")
    return YOLO("yolov8n.pt")

model = load_yolo_model()


# ─────────────────────────────────────────────
# Helper: Road Priority Classification
# ─────────────────────────────────────────────
def classify_road_priority(score, severe_count, total_count):
    """Return (priority_level, badge_class, action_text)"""
    if score < 40 or severe_count >= 3:
        return "🔴 CRITICAL", "badge-critical", "Immediate repair required — road unsafe for vehicles!"
    elif score < 60 or severe_count >= 1:
        return "🟠 HIGH", "badge-high", "Schedule repair within 7 days — significant damage present."
    elif score < 80 or total_count >= 3:
        return "🟡 MEDIUM", "badge-medium", "Plan maintenance within 30 days — moderate wear detected."
    else:
        return "🟢 LOW", "badge-low", "Road in good condition — routine monitoring recommended."


def get_health_bar_color(score):
    if score >= 80:
        return "linear-gradient(90deg,#22c55e,#4ade80)"
    elif score >= 60:
        return "linear-gradient(90deg,#eab308,#facc15)"
    elif score >= 40:
        return "linear-gradient(90deg,#f97316,#fb923c)"
    else:
        return "linear-gradient(90deg,#ef4444,#f87171)"


# ─────────────────────────────────────────────
# Helper: Run Detection on a PIL Image
# ─────────────────────────────────────────────
def run_detection(rgb_image_pil, disparity_arr=None):
    """
    Run YOLOv8 on rgb_image_pil. 
    If disparity_arr is None, a dummy flat array is created (depth=0, area-only severity).
    Returns: draw_image, pothole_list, severities
    """
    img_arr = np.array(rgb_image_pil.convert("RGB"))

    # If no disparity map provided, create dummy (flat 128)
    H, W = img_arr.shape[:2]
    if disparity_arr is None:
        disparity_arr = np.full((H, W), 128, dtype=np.uint8)

    results = model(img_arr, verbose=False)
    boxes = results[0].boxes

    draw_image = rgb_image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(draw_image)

    try:
        font_large = ImageFont.load_default()
    except Exception:
        font_large = None

    pothole_list = []
    severities = []

    color_map = {
        "Mild":     ("#eab308", (234, 179,  8)),
        "Moderate": ("#f97316", (249, 115, 22)),
        "Severe":   ("#ef4444", (239,  68, 68)),
    }

    for i, box in enumerate(boxes):
        coords = box.xyxy[0].cpu().numpy()
        xmin, ymin, xmax, ymax = coords
        conf = float(box.conf[0])

        metrics = estimate_pothole_metrics([xmin, ymin, xmax, ymax], disparity_arr)
        severity = metrics["severity"]
        severities.append(severity)

        _, rgb_col = color_map.get(severity, ("#ef4444", (239, 68, 68)))

        # Draw rounded-corner rectangle (draw 3px thick)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=rgb_col, width=3)

        # Draw label background
        label = f"#{i+1} {severity}  {conf:.0%}"
        bbox_text = draw.textbbox((xmin + 4, ymin + 4), label, font=font_large)
        draw.rectangle(
            [bbox_text[0]-3, bbox_text[1]-2, bbox_text[2]+3, bbox_text[3]+2],
            fill=rgb_col
        )
        draw.text((xmin + 4, ymin + 4), label, fill=(255, 255, 255), font=font_large)

        pothole_list.append({
            "ID": i + 1,
            "Severity": severity,
            "Priority": "🔴 Severe" if severity == "Severe" else ("🟠 Moderate" if severity == "Moderate" else "🟡 Mild"),
            "Width (px)": metrics["width"],
            "Height (px)": metrics["height"],
            "Area (px²)": metrics["bbox_area"],
            "Depth Score": f"{metrics['depth_score']:.2f}",
            "Confidence": f"{conf:.1%}",
            "Bbox": [xmin, ymin, xmax, ymax]
        })

    return draw_image, pothole_list, severities


# ─────────────────────────────────────────────
# Helper: Render Metrics Row
# ─────────────────────────────────────────────
def render_metrics(health_score, pothole_list, severities, frame_label=""):
    severe_count  = severities.count("Severe")
    mod_count     = severities.count("Moderate")
    mild_count    = severities.count("Mild")
    total         = len(pothole_list)
    priority, badge_cls, action_text = classify_road_priority(health_score, severe_count, total)
    bar_color = get_health_bar_color(health_score)

    if health_score >= 80:  hc = "#22c55e"
    elif health_score >= 60: hc = "#eab308"
    elif health_score >= 40: hc = "#f97316"
    else:                    hc = "#ef4444"

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Road Health Score</div>
        <div class="metric-val" style="color:{hc};">{health_score}<span style="font-size:18px;color:#64748b;">/100</span></div>
        <div class="health-bar-container">
            <div class="health-bar-fill" style="width:{health_score}%;background:{bar_color};"></div>
        </div>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Priority Level</div>
        <div class="metric-val" style="font-size:20px;margin-top:10px;">{priority}</div>
        <div class="metric-sub" style="color:#94a3b8;font-size:11px;margin-top:6px;">{action_text[:50]}…</div>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Potholes Detected</div>
        <div class="metric-val" style="color:#38bdf8;">{total}</div>
        <div class="metric-sub">🔴 {severe_count}  🟠 {mod_count}  🟡 {mild_count}</div>
    </div>""", unsafe_allow_html=True)

    c4.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Severe Potholes</div>
        <div class="metric-val" style="color:#ef4444;">{severe_count}</div>
        <div class="metric-sub">Immediate attention needed</div>
    </div>""", unsafe_allow_html=True)

    c5.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Frame / Source</div>
        <div class="metric-val" style="color:#cbd5e1;font-size:20px;margin-top:10px;">{frame_label}</div>
    </div>""", unsafe_allow_html=True)

    # Priority action alert
    alert_colors = {
        "badge-critical": ("#7f1d1d", "#ef4444"),
        "badge-high":     ("#7c2d12", "#f97316"),
        "badge-medium":   ("#713f12", "#eab308"),
        "badge-low":      ("#14532d", "#22c55e"),
    }
    bg, border = alert_colors.get(badge_cls, ("#1e293b", "#6366f1"))
    st.markdown(f"""
    <div style="background:rgba({','.join(str(int(bg[i:i+2],16)) for i in (1,3,5))},0.25);
                border-left:4px solid {border};border-radius:8px;padding:14px 20px;margin:12px 0;">
        <b style="color:{border};">{priority}</b>&nbsp;&nbsp;
        <span style="color:#e2e8f0;">{action_text}</span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: Render GPS Map
# ─────────────────────────────────────────────
def render_gps_map(pothole_list, frame_idx=0, base_lat=12.9716, base_lon=77.5946):
    vehicle_lat = base_lat + 0.00012 * frame_idx
    vehicle_lon = base_lon + 0.00008 * np.sin(frame_idx / 8.0)

    m = folium.Map(location=[vehicle_lat, vehicle_lon], zoom_start=18, tiles='CartoDB dark_matter')

    # Trail
    trail = [[base_lat + 0.00012*i, base_lon + 0.00008*np.sin(i/8.0)] for i in range(frame_idx+1)]
    if len(trail) > 1:
        folium.PolyLine(trail, color="#6366f1", weight=3, opacity=0.8, tooltip="Inspection Route").add_to(m)

    # Vehicle marker
    folium.Marker(
        [vehicle_lat, vehicle_lon],
        popup=f"Inspection Vehicle (Frame {frame_idx})",
        tooltip="🚗 Vehicle",
        icon=folium.Icon(color="blue", icon="car", prefix="fa")
    ).add_to(m)

    # Pothole markers sorted by severity (severe first = highest priority)
    sev_order = {"Severe": 0, "Moderate": 1, "Mild": 2}
    sorted_ph = sorted(pothole_list, key=lambda x: sev_order.get(x["Severity"], 3))

    for ph in sorted_ph:
        bbox = ph["Bbox"]
        x_center = (bbox[0] + bbox[2]) / 2.0
        offset_lat = vehicle_lat + 0.00003 + 0.00001 * (x_center - 200) / 200.0
        offset_lon = vehicle_lon + 0.00002
        sev = ph["Severity"]
        color_mk = "red" if sev == "Severe" else ("orange" if sev == "Moderate" else "beige")
        icon_name = "exclamation-triangle" if sev == "Severe" else ("exclamation" if sev == "Moderate" else "info")

        folium.Marker(
            [offset_lat, offset_lon],
            popup=folium.Popup(
                f"<b>Pothole #{ph['ID']}</b><br>"
                f"Severity: <b>{sev}</b><br>"
                f"Priority: {ph['Priority']}<br>"
                f"Area: {ph['Area (px²)']} px²<br>"
                f"Depth Score: {ph['Depth Score']}<br>"
                f"Confidence: {ph['Confidence']}",
                max_width=200
            ),
            tooltip=f"#{ph['ID']} {sev}",
            icon=folium.Icon(color=color_mk, icon=icon_name, prefix="fa")
        ).add_to(m)

    st.components.v1.html(m._repr_html_(), height=440)


# ─────────────────────────────────────────────
# Helper: Render Damage Table
# ─────────────────────────────────────────────
def render_damage_table(pothole_list):
    st.markdown("### 📋 Damage Inventory (Priority Order)")
    if len(pothole_list) == 0:
        st.success("✅ No potholes detected — road surface appears healthy!")
        return
    sev_order = {"Severe": 0, "Moderate": 1, "Mild": 2}
    df = pd.DataFrame(pothole_list).drop(columns=["Bbox"])
    df = df.sort_values("Severity", key=lambda col: col.map(sev_order))

    # Color rows by severity
    def highlight_severity(row):
        colors = {"Severe": "background-color:#4b1d1d;color:#fca5a5",
                  "Moderate": "background-color:#431a05;color:#fdba74",
                  "Mild": "background-color:#422006;color:#fde68a"}
        style = colors.get(row["Severity"], "")
        return [style] * len(row)

    st.dataframe(df.style.apply(highlight_severity, axis=1), use_container_width=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.markdown("# 🚧 Road Inspection")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🎯 Inspection Mode")
mode = st.sidebar.radio(
    "Select Mode",
    ["📂 Dataset Browse", "📸 Live Photo / Upload"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Severity Guide")
st.sidebar.markdown("""
| Level | Deduction | Threshold |
|-------|-----------|-----------|
| 🟡 Mild | -5 pts | Area < 1200px² |
| 🟠 Moderate | -10 pts | Area 1200-3500px² |
| 🔴 Severe | -20 pts | Area > 3500px² |
""")

st.sidebar.markdown("### 🚦 Priority Guide")
st.sidebar.markdown("""
- 🔴 **Critical** — Score < 40 or ≥ 3 severe  
- 🟠 **High** — Score < 60 or ≥ 1 severe  
- 🟡 **Medium** — Score < 80 or ≥ 3 potholes  
- 🟢 **Low** — Road in good condition
""")

# ─────────────────────────────────────────────
# Dashboard Header
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:20px 0 10px;">
    <h1 style="background:linear-gradient(135deg,#6366f1,#8b5cf6,#06b6d4);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               font-size:2.4rem;font-weight:800;margin-bottom:4px;">
        🚧 Municipal Road Damage Inspection
    </h1>
    <p style="color:#94a3b8;font-size:1rem;">
        Automated pothole detection & severity estimation using <b style="color:#8b5cf6;">YOLOv8-Nano</b>
        — with priority-based road health scoring & live GPS mapping
    </p>
</div>
<hr class="section-divider">
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# MODE 1 — DATASET BROWSE
# ═══════════════════════════════════════════════
if mode == "📂 Dataset Browse":
    base_dir      = r"c:\Users\meher\Downloads\mini DATASET"
    test_rgb_dir  = os.path.join(base_dir, "archive", "pothole600", "testing", "rgb")
    test_disp_dir = os.path.join(base_dir, "archive", "pothole600", "testing", "tdisp")

    if os.path.exists(test_rgb_dir):
        test_files = sorted([f for f in os.listdir(test_rgb_dir) if f.lower().endswith(".png")])
    else:
        test_files = []

    if len(test_files) == 0:
        st.error("❌ No test images found. Verify dataset path: `archive/pothole600/testing/rgb/`")
        st.stop()

    frame_idx = st.sidebar.slider("Road Frame Index", 0, len(test_files)-1, 0)

    file_name = test_files[frame_idx]
    rgb_path  = os.path.join(test_rgb_dir, file_name)
    disp_path = os.path.join(test_disp_dir, file_name)

    if not os.path.exists(rgb_path):
        st.error(f"Missing RGB file: {file_name}")
        st.stop()

    rgb_pil = Image.open(rgb_path)
    disp_arr = None
    if os.path.exists(disp_path):
        disp_arr = np.array(Image.open(disp_path).convert("L"))

    with st.spinner("🔍 Running YOLOv8 detection…"):
        draw_image, pothole_list, severities = run_detection(rgb_pil, disp_arr)

    health_score = calculate_road_health_score(severities)
    render_metrics(health_score, pothole_list, severities, frame_label=f"#{frame_idx:04d}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("### 📷 RGB — Detections")
        st.image(draw_image, use_container_width=True,
                 caption=f"YOLOv8 detections — {file_name}")
    with col_img2:
        st.markdown("### 🛰️ Disparity / Depth Map")
        if disp_arr is not None:
            colorized = cv2.applyColorMap(disp_arr, cv2.COLORMAP_PLASMA)
            st.image(colorized, use_container_width=True,
                     caption="Stereo disparity map (Yellow=near, Purple=far)")
        else:
            st.info("No disparity map available for this frame.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    col_table, col_map = st.columns(2)
    with col_table:
        render_damage_table(pothole_list)
    with col_map:
        st.markdown("### 🗺️ Live GPS Damage Map")
        render_gps_map(pothole_list, frame_idx=frame_idx)


# ═══════════════════════════════════════════════
# MODE 2 — LIVE PHOTO / UPLOAD
# ═══════════════════════════════════════════════
else:
    st.markdown("## 📸 Live Photo Inspection")
    st.markdown("Upload a road photo **or** snap one with your webcam to instantly detect potholes and get a road health score.")

    tab_upload, tab_camera = st.tabs(["📁 Upload Photo", "📷 Webcam Capture"])

    image_source = None
    source_label = "Uploaded Photo"

    # ── Tab 1: File Upload ──
    with tab_upload:
        uploaded = st.file_uploader(
            "Drop a road photo here (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            help="Best results with clear daylight road images taken from a vehicle-mounted or handheld camera."
        )
        if uploaded:
            image_source = Image.open(uploaded)
            source_label = uploaded.name

    # ── Tab 2: Webcam ──
    with tab_camera:
        st.info("📷 Allow browser camera access, then click **Take Photo** to capture a live road image.")
        cam_photo = st.camera_input("Take a road photo")
        if cam_photo:
            image_source = Image.open(cam_photo)
            source_label = "Webcam Capture"

    # ── Process if we have an image ──
    if image_source is not None:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("### 🖼️ Original Photo")
            st.image(image_source, use_container_width=True, caption=source_label)

        with st.spinner("🔍 Analyzing road damage with YOLOv8-Nano…"):
            # No disparity map for live photos — area-based severity only
            draw_image, pothole_list, severities = run_detection(image_source, disparity_arr=None)
            health_score = calculate_road_health_score(severities)

        with col_det:
            st.markdown("### 🚧 Detected Damage")
            st.image(draw_image, use_container_width=True,
                     caption=f"Potholes detected: {len(pothole_list)}")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Metrics
        render_metrics(health_score, pothole_list, severities, frame_label="📸 Live")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Download annotated image
        buf = io.BytesIO()
        draw_image.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Annotated Image",
            data=buf.getvalue(),
            file_name=f"road_damage_{source_label.replace(' ','_')}.png",
            mime="image/png"
        )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        col_table, col_map = st.columns(2)
        with col_table:
            render_damage_table(pothole_list)
        with col_map:
            st.markdown("### 🗺️ GPS Damage Map (Simulated)")
            st.caption("GPS coordinates are simulated. Integrate a real GPS feed for live tracking.")
            render_gps_map(pothole_list, frame_idx=0)

    else:
        # Placeholder when no image is loaded yet
        st.markdown("""
        <div style="
            background:rgba(30,41,59,0.6);
            border:2px dashed rgba(99,102,241,0.4);
            border-radius:20px;
            padding:60px 40px;
            text-align:center;
            margin:30px 0;
        ">
            <div style="font-size:56px;margin-bottom:16px;">📸</div>
            <h3 style="color:#8b5cf6;margin-bottom:8px;">Ready for Road Inspection</h3>
            <p style="color:#64748b;font-size:15px;">
                Upload a photo using the <b>📁 Upload Photo</b> tab<br>
                or snap one live using the <b>📷 Webcam Capture</b> tab above.
            </p>
            <div style="margin-top:20px;display:flex;justify-content:center;gap:20px;flex-wrap:wrap;">
                <span style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);
                             border-radius:999px;padding:6px 16px;color:#a5b4fc;font-size:13px;">
                    🔍 YOLOv8-Nano Inference
                </span>
                <span style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);
                             border-radius:999px;padding:6px 16px;color:#a5b4fc;font-size:13px;">
                    📊 Severity Scoring
                </span>
                <span style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);
                             border-radius:999px;padding:6px 16px;color:#a5b4fc;font-size:13px;">
                    🗺️ GPS Mapping
                </span>
                <span style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);
                             border-radius:999px;padding:6px 16px;color:#a5b4fc;font-size:13px;">
                    ⬇️ Download Report
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
