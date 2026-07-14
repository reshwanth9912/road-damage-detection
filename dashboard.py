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
    # 1. Try path from Streamlit Secrets (if configured)
    # 2. Try relative path within repository
    # 3. Try hardcoded local absolute path
    paths_to_try = []
    try:
        if "MODEL_PATH" in st.secrets:
            paths_to_try.append(st.secrets["MODEL_PATH"])
    except Exception:
        pass
    
    paths_to_try.append(os.path.join(os.path.dirname(__file__), "models", "best.pt"))
    paths_to_try.append(r"c:\Users\meher\Downloads\mini DATASET\models\best.pt")
    
    for path in paths_to_try:
        if path and os.path.exists(path):
            return YOLO(path)
            
    st.warning("⚠️ Trained model not found — using default YOLO11-Nano weights.")
    return YOLO("yolo11n.pt")

model = load_yolo_model()


# ─────────────────────────────────────────────
# Helper: Road Priority Classification
# ─────────────────────────────────────────────
def classify_road_priority(score, severe_count, total_count):
    """Return (priority_level, badge_class, action_text)"""
    if total_count == 0 and score == 0:
        return "⚪ PENDING", "badge-low", "Upload or capture a photo to begin inspection."
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
def run_detection(rgb_image_pil, disparity_arr=None, manual_conf=None):
    """
    Run YOLO11 on rgb_image_pil.
    Uses YOLO's built-in NMS. Each pothole is detected individually and numbered.
    Returns: draw_image, pothole_list, severities
    """
    img_arr = np.array(rgb_image_pil.convert("RGB"))

    # If no disparity map provided, create dummy (flat 128)
    H, W = img_arr.shape[:2]
    if disparity_arr is None:
        disparity_arr = np.full((H, W), 128, dtype=np.uint8)

    if manual_conf is not None:
        # User specified manual confidence threshold
        dynamic_conf = manual_conf
        results = model(img_arr, conf=dynamic_conf, iou=0.3, imgsz=640, verbose=False)
        boxes = results[0].boxes
        final_boxes = []
        for box in boxes:
            conf = float(box.conf[0])
            coords = box.xyxy[0].cpu().numpy()
            final_boxes.append([float(coords[0]), float(coords[1]),
                                float(coords[2]), float(coords[3]), conf])
    else:
        # ── Auto-calibration logic ──
        # Run initial inference with extremely low confidence to see candidate landscape
        results = model(img_arr, conf=0.01, iou=0.3, imgsz=640, verbose=False)
        boxes = results[0].boxes

        confs = [float(b.conf[0]) for b in boxes]
        solid_candidates = [c for c in confs if c >= 0.02]

        # Choose dynamic confidence:
        # If there are many potential detections, we are on a road segment with multiple potholes.
        # Otherwise, we are looking at a clean road or a road with only a single clear pothole.
        if len(solid_candidates) >= 8:
            dynamic_conf = 0.01  # highly sensitive to catch all 15-20 potholes
        else:
            dynamic_conf = 0.30  # high threshold to isolate the single major pothole and avoid noise

        final_boxes = []
        for box in boxes:
            conf = float(box.conf[0])
            if conf >= dynamic_conf:
                coords = box.xyxy[0].cpu().numpy()
                final_boxes.append([float(coords[0]), float(coords[1]),
                                    float(coords[2]), float(coords[3]), conf])

    # Sort by Y position (top-to-bottom) then X (left-to-right) for natural numbering
    final_boxes.sort(key=lambda b: (b[1], b[0]))

    # ── Draw detections ──
    draw_image = rgb_image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(draw_image)

    # Try to load fonts for better visibility
    font_label = ImageFont.load_default()
    try:
        font_label = ImageFont.truetype("arial.ttf", size=max(13, int(min(H, W) * 0.020)))
    except Exception:
        try:
            font_label = ImageFont.truetype("arialbd.ttf", size=max(13, int(min(H, W) * 0.020)))
        except Exception:
            font_label = ImageFont.load_default()

    font_number = ImageFont.load_default()
    try:
        font_number = ImageFont.truetype("arialbd.ttf", size=max(16, int(min(H, W) * 0.026)))
    except Exception:
        try:
            font_number = ImageFont.truetype("arial.ttf", size=max(16, int(min(H, W) * 0.026)))
        except Exception:
            font_number = ImageFont.load_default()

    pothole_list = []
    severities = []

    color_map = {
        "Mild":     ("#eab308", (234, 179,   8)),
        "Moderate": ("#f97316", (249, 115,  22)),
        "Severe":   ("#ef4444", (239,  68,  68)),
    }

    for i, box in enumerate(final_boxes):
        xmin, ymin, xmax, ymax, conf = box

        metrics = estimate_pothole_metrics([xmin, ymin, xmax, ymax], disparity_arr)
        severity = metrics["severity"]
        severities.append(severity)

        _, rgb_col = color_map.get(severity, ("#ef4444", (239, 68, 68)))

        # Draw bounding box border (3px thick)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=rgb_col, width=3)

        # ── Number tag (circle with number) at top-left corner ──
        tag_text = str(i + 1)
        tag_bbox = draw.textbbox((0, 0), tag_text, font=font_number)
        tag_w = tag_bbox[2] - tag_bbox[0]
        tag_h = tag_bbox[3] - tag_bbox[1]
        tag_radius = max(tag_w, tag_h) // 2 + 6
        tag_cx = int(xmin) + tag_radius + 2
        tag_cy = int(ymin) - tag_radius - 2  # Place above the box
        # If tag would go off-screen, place inside the box
        if tag_cy - tag_radius < 0:
            tag_cy = int(ymin) + tag_radius + 4

        # Draw circle background for number
        draw.ellipse(
            [tag_cx - tag_radius, tag_cy - tag_radius,
             tag_cx + tag_radius, tag_cy + tag_radius],
            fill=rgb_col, outline=(255, 255, 255), width=2
        )
        # Draw number centered in circle
        draw.text(
            (tag_cx - tag_w // 2, tag_cy - tag_h // 2 - 1),
            tag_text, fill=(255, 255, 255), font=font_number
        )

        # ── Severity + Confidence label at bottom of the box ──
        label = f"{severity} {conf:.0%}"
        lbl_bbox = draw.textbbox((xmin, ymax + 2), label, font=font_label)
        lbl_w = lbl_bbox[2] - lbl_bbox[0]
        lbl_h = lbl_bbox[3] - lbl_bbox[1]
        # If label goes off-screen at bottom, place inside the box at the top
        if ymax + 2 + lbl_h > H:
            lbl_y = int(ymin) + 4
        else:
            lbl_y = int(ymax) + 2
        lbl_bbox = draw.textbbox((xmin + 2, lbl_y), label, font=font_label)
        draw.rectangle(
            [lbl_bbox[0] - 2, lbl_bbox[1] - 1, lbl_bbox[2] + 2, lbl_bbox[3] + 1],
            fill=rgb_col
        )
        draw.text((xmin + 2, lbl_y), label, fill=(255, 255, 255), font=font_label)

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
# ─────────────────────────────────────────────
# Helper: Render Metrics Row
# ─────────────────────────────────────────────
def render_metrics(health_score, pothole_list, severities, frame_label=""):
    severe_count  = severities.count("Severe")
    mod_count     = severities.count("Moderate")
    mild_count    = severities.count("Mild")
    total         = len(pothole_list)
    
    if total == 0 and health_score == 0:
        health_score_display = "--"
        hc = "#cbd5e1"
        bar_color = "rgba(100,116,139,0.3)"
        priority = "⚪ PENDING"
        action_text = "Click 'Run Road Inspection' to analyze this frame."
        badge_cls = "badge-low"
    else:
        health_score_display = str(health_score)
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
        <div class="metric-val" style="color:{hc};">{health_score_display}<span style="font-size:18px;color:#64748b;">/100</span></div>
        <div class="health-bar-container">
            <div class="health-bar-fill" style="width:{health_score if health_score_display != '--' else 0}%;background:{bar_color};"></div>
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
    [" Live Photo / Upload", "📹 Live Camera Stream"],
    index=1,
    label_visibility="collapsed"
)

st.sidebar.markdown('<p style="font-size:16px;font-weight:bold;margin-bottom:2px;margin-top:10px;">🛠️ Detection Settings</p>', unsafe_allow_html=True)
sens_mode = st.sidebar.radio(
    "Sensitivity Mode",
    options=["✨ Auto-Calibrate", "🎛️ manual threshold"],
    index=0,
    help="Auto-Calibrate dynamically adjusts sensitivity for single vs. multi-pothole roads."
)

manual_conf = None
if sens_mode == "🎛️ Manual Threshold":
    manual_conf = st.sidebar.slider("Confidence Cutoff", 0.01, 0.50, 0.15, step=0.01)

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
        Automated pothole detection & severity estimation using <b style="color:#8b5cf6;">YOLO11-Nano</b>
        — with priority-based road health scoring & live GPS mapping
    </p>
</div>
<hr class="section-divider">
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# MODE 2 — LIVE PHOTO / UPLOAD
# ═══════════════════════════════════════════════
if mode == "📸 Live Photo / Upload":
    st.markdown("## 📸 Live Photo Inspection")
    st.markdown("Upload a road photo **or** snap one with your webcam to instantly detect potholes and get a road health score.")

    tab_upload, tab_camera = st.tabs(["📁 Upload Photo", "📷 Webcam Snapshot"])

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

    # ── Tab 2: Webcam Snapshot ──
    with tab_camera:
        st.info("📷 Allow browser camera access, then click **Take Photo** to capture a single road image.")
        cam_photo = st.camera_input("Take a road photo")
        if cam_photo:
            image_source = Image.open(cam_photo)
            source_label = "Webcam Snapshot"

    # ── Process if we have an image ──
    if image_source is not None:
        if "last_source" not in st.session_state or st.session_state.last_source != source_label:
            st.session_state.last_source = source_label
            st.session_state.inspected_live = False
            st.session_state.live_results = None

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        st.sidebar.markdown('<hr style="border:1px solid rgba(99,102,241,0.25);margin:10px 0;">', unsafe_allow_html=True)
        if st.sidebar.button("🔍 Run Road Inspection", key="btn_run_live", use_container_width=True):
            with st.spinner("🔍 Analyzing road damage dynamically with YOLO11-Nano…"):
                draw_image, pothole_list, severities = run_detection(image_source, disparity_arr=None, manual_conf=manual_conf)
                health_score = calculate_road_health_score(severities)
                st.session_state.live_results = {
                    "draw_image": draw_image,
                    "pothole_list": pothole_list,
                    "severities": severities,
                    "health_score": health_score
                }
                st.session_state.inspected_live = True

        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("### 🖼️ Original Photo")
            st.image(image_source, use_container_width=True, caption=source_label)

        if st.session_state.get("inspected_live") and st.session_state.get("live_results") is not None:
            res_live = st.session_state.live_results
            with col_det:
                st.markdown("### 🚧 Detected Damage")
                st.image(res_live["draw_image"], use_container_width=True,
                         caption=f"Potholes detected: {len(res_live['pothole_list'])}")

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            render_metrics(res_live["health_score"], res_live["pothole_list"], res_live["severities"], frame_label="📸 Live")
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            buf = io.BytesIO()
            res_live["draw_image"].save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Annotated Image",
                data=buf.getvalue(),
                file_name=f"road_damage_{source_label.replace(' ','_')}.png",
                mime="image/png"
            )

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            col_table, col_map = st.columns(2)
            with col_table:
                render_damage_table(res_live["pothole_list"])
            with col_map:
                st.markdown("### 🗺️ GPS Damage Map (Simulated)")
                st.caption("GPS coordinates are simulated.")
                render_gps_map(res_live["pothole_list"], frame_idx=0)
        else:
            with col_det:
                st.markdown("### 🚧 Detected Damage")
                st.info("Awaiting inspection. Click **Run Road Inspection** in the sidebar to begin.")
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            render_metrics(0, [], [], frame_label="📸 Live (Pending)")
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.info("💡 Click **Run Road Inspection** in the sidebar to run YOLO11 damage analysis on this image.")

    else:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        render_metrics(0, [], [], frame_label="Awaiting Input")
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(30,41,59,0.6);border:2px dashed rgba(99,102,241,0.4);
                    border-radius:20px;padding:60px 40px;text-align:center;margin:30px 0;">
            <div style="font-size:56px;margin-bottom:16px;">📸</div>
            <h3 style="color:#8b5cf6;margin-bottom:8px;">Ready for Road Inspection</h3>
            <p style="color:#64748b;font-size:15px;">
                Upload a photo using the <b>📁 Upload Photo</b> tab<br>
                or snap one live using the <b>📷 Webcam Snapshot</b> tab above.
            </p>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# MODE 3 — LIVE CAMERA STREAM
# ═══════════════════════════════════════════════
else:
    st.markdown("## 📹 Live Camera Stream")
    st.markdown("Stream from your **local webcam** in real-time. YOLO11 detects potholes on every captured frame.")

    # ── Session state init ──
    if "cam_running" not in st.session_state:
        st.session_state.cam_running = True

    st.sidebar.markdown('<hr style="border:1px solid rgba(99,102,241,0.25);margin:10px 0;">', unsafe_allow_html=True)
    run_detection_live = st.sidebar.checkbox("🔍 Run YOLO11 Detection", value=True,
                                             help="Uncheck to see raw feed without inference (faster).")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av as av_lib
        import threading

        RTC_CONFIG = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        class PotholeDetector(VideoProcessorBase):
            def __init__(self):
                self.run_det = True
                self.pothole_count = 0
                self.health_score = 100
                self.severities = []
                self.lock = threading.Lock()

            def recv(self, frame):
                img_bgr = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

                if self.run_det:
                    try:
                        det_pil, pothole_list, sevs = run_detection(
                            img_pil, disparity_arr=None, manual_conf=manual_conf
                        )
                        hs = calculate_road_health_score(sevs)
                        with self.lock:
                            self.pothole_count = len(pothole_list)
                            self.health_score = hs
                            self.severities = list(sevs)
                        out_bgr = cv2.cvtColor(np.array(det_pil), cv2.COLOR_RGB2BGR)
                    except Exception:
                        out_bgr = img_bgr
                else:
                    with self.lock:
                        self.pothole_count = 0
                        self.health_score = 100
                        self.severities = []
                    out_bgr = img_bgr

                return av_lib.VideoFrame.from_ndarray(out_bgr, format="bgr24")

        col_feed, col_info = st.columns([3, 2])
        with col_feed:
            st.markdown("### 📹 Live Camera Feed")
            st.caption("Click **START** below the video, then allow camera access in the browser popup.")
            ctx = webrtc_streamer(
                key="pothole-live-cam",
                video_processor_factory=PotholeDetector,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
            )

        with col_info:
            st.markdown("### 📊 Live Detection Info")
            if ctx and ctx.video_processor:
                ctx.video_processor.run_det = run_detection_live
                with ctx.video_processor.lock:
                    total    = ctx.video_processor.pothole_count
                    hs       = ctx.video_processor.health_score
                    sevs     = ctx.video_processor.severities
                severe_c = sevs.count("Severe")
                mod_c    = sevs.count("Moderate")
                mild_c   = sevs.count("Mild")

                if hs >= 80:   hc = "#22c55e"
                elif hs >= 60: hc = "#eab308"
                elif hs >= 40: hc = "#f97316"
                else:          hc = "#ef4444"

                st.markdown(f"""
                <div style="display:flex;flex-direction:column;gap:12px;margin-top:12px;">
                    <div class="metric-card">
                        <div class="metric-label">Road Health Score</div>
                        <div class="metric-val" style="color:{hc};">{hs}<span style="font-size:18px;color:#64748b;">/100</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Potholes Detected</div>
                        <div class="metric-val" style="color:#38bdf8;">{total}</div>
                        <div class="metric-sub">🔴 {severe_c} &nbsp; 🟠 {mod_c} &nbsp; 🟡 {mild_c}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(
                    "� **Camera not yet started.**\n\n"
                    "1. Click the **START** button in the video box\n"
                    "2. Allow camera access when browser asks\n"
                    "3. YOLO11 will run on every frame automatically\n\n"
                    "Toggle **🔍 Run YOLO11 Detection** in the sidebar to switch between raw and detected feed."
                )

    except ImportError:
        st.error("❌ `streamlit-webrtc` not installed. Run: `pip install streamlit-webrtc av`")
