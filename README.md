<div align="center">

# 🚧 Municipal Road Damage Inspection & Mapping System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![YOLO11](https://img.shields.io/badge/YOLO11-Nano-00FFAA?style=for-the-badge)](https://ultralytics.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**Real-time pothole detection, severity estimation, and GPS mapping for smart municipal maintenance.**

[🚀 Live Demo](#-quick-start) · [📸 Features](#-features) · [📦 Installation](#-installation) · [🗺 Deployment](#-deployment)

</div>

---

## 📌 Overview

A complete end-to-end **road damage detection and severity estimation system** built for municipal infrastructure teams. Using a custom-trained **YOLO11-Nano** model on the **Pothole600 dataset**, the system automatically:

- Detects and numbers individual potholes in images and live video
- Estimates severity (Mild / Moderate / Severe) based on area and depth
- Calculates a **Road Health Score (0–100)** for maintenance prioritization
- Maps pothole locations on an interactive GPS map
- Streams live webcam feed with real-time YOLO11 inference via **WebRTC**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **YOLO11-Nano Inference** | Custom-trained on Pothole600 — fast, accurate detection |
| 📂 **Dataset Browse Mode** | Navigate 600+ test frames with disparity depth maps |
| 📸 **Photo Upload / Webcam Snapshot** | Upload or snap a road photo for instant analysis |
| 📹 **Live Camera Stream** | Real-time pothole detection via WebRTC browser camera |
| 📊 **Road Health Score** | 0–100 score with severity breakdown and priority badge |
| 🗺️ **GPS Damage Map** | Folium-based interactive map with pothole markers |
| ⚡ **Adaptive Confidence Threshold** | Auto-calibrates for single vs. multi-pothole scenes |
| 🎛️ **Manual Confidence Override** | Fine-tune detection sensitivity via sidebar slider |
| ⬇️ **Annotated Image Export** | Download detection results as PNG |

---

## 🖼️ Screenshots

> Dashboard running in Live Camera Stream mode with real-time YOLO11 detections and Road Health Score.

---

## 🧠 Model Architecture

| Component | Details |
|---|---|
| **Model** | YOLO11-Nano (`yolo11n.pt` base, fine-tuned) |
| **Dataset** | Pothole600 (RGB + stereo disparity maps) |
| **Task** | Object Detection (pothole bounding boxes) |
| **Input Size** | 640 × 640 |
| **Severity Logic** | Area-based: Mild < 1200px², Moderate 1200–3500px², Severe > 3500px² |
| **Health Score** | 100 − (5×Mild + 10×Moderate + 20×Severe) |

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- Webcam (for Live Camera Stream mode)

### 1. Clone the repository
```bash
git clone https://github.com/reshwanth9912/road-damage-detection.git
cd road-damage-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset (optional)
Place the **Pothole600** dataset under:
```
archive/pothole600/testing/rgb/       ← RGB test images (.png)
archive/pothole600/testing/tdisp/     ← Disparity maps (.png)
```

### 4. Place the trained model
```
models/best.pt
```
> If `models/best.pt` is not found, the system falls back to default YOLO11-Nano weights.

---

## 🚀 Quick Start

```bash
streamlit run dashboard.py
```

Open your browser at **http://localhost:8501**

### Modes
| Mode | How to use |
|---|---|
| 📂 Dataset Browse | Use the frame slider → click **Run Road Inspection** |
| 📸 Live Photo / Upload | Upload a road image or use Webcam Snapshot tab |
| 📹 Live Camera Stream | Click **START** → allow browser camera access → detection runs live |

---

## 📁 Project Structure

```
road-damage-detection/
├── dashboard.py            # Main Streamlit application
├── severity_estimator.py   # Pothole metrics & Road Health Score logic
├── train_yolo.py           # YOLO11 training script
├── detect_all.py           # Batch inference on test set
├── prepare_dataset.py      # Dataset preparation utilities
├── scan_potholes.py        # Quick single-image scan script
├── inspect_data.py         # Dataset inspection utilities
├── find_matching_frame.py  # Frame matching utility
├── tests/                  # Unit tests
├── models/                 # Trained model weights (best.pt)
├── archive/                # Pothole600 dataset
│   └── pothole600/
│       └── testing/
│           ├── rgb/        # RGB road images
│           └── tdisp/      # Stereo disparity maps
├── requirements.txt
└── README.md
```

---

## 🗺 Deployment

### Streamlit Community Cloud (Recommended — Free)
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New App → select your repo → set `dashboard.py` → Deploy
> Live camera works on Cloud via WebRTC (TURN server configured automatically).

### Render / Railway
Set the start command:
```bash
streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0
```

---

## 🔧 Configuration

All detection parameters can be tuned live in the sidebar:

| Setting | Options | Effect |
|---|---|---|
| **Sensitivity Mode** | Auto-Calibrate / Manual Threshold | Auto adjusts for scene complexity |
| **Confidence Cutoff** | 0.01 – 0.50 | Controls detection sensitivity |
| **Run YOLO11 Detection** | On / Off (Live mode) | Toggle inference on live feed |

---

## 🧪 Severity & Priority Guide

### Severity Levels
| Level | Area Threshold | Health Deduction |
|---|---|---|
| 🟡 Mild | < 1200 px² | −5 pts |
| 🟠 Moderate | 1200 – 3500 px² | −10 pts |
| 🔴 Severe | > 3500 px² | −20 pts |

### Priority Levels
| Priority | Condition | Action |
|---|---|---|
| 🔴 Critical | Score < 40 or ≥ 3 severe | Immediate repair |
| 🟠 High | Score < 60 or ≥ 1 severe | Repair within 7 days |
| 🟡 Medium | Score < 80 or ≥ 3 potholes | Plan within 30 days |
| 🟢 Low | Score ≥ 80 | Routine monitoring |

---

## 🛠️ Tech Stack

- **[Ultralytics YOLO11](https://ultralytics.com)** — Object detection backbone
- **[Streamlit](https://streamlit.io)** — Interactive web dashboard
- **[streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)** — Browser-based live camera streaming
- **[OpenCV](https://opencv.org)** — Image processing
- **[Folium](https://python-visualization.github.io/folium/)** — GPS map rendering
- **[Pillow](https://pillow.readthedocs.io/)** — Image annotation
- **[Pandas](https://pandas.pydata.org/)** — Damage inventory table

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Reshwanth** · [@reshwanth9912](https://github.com/reshwanth9912)

> Built for smart city municipal infrastructure management using AI-powered road inspection.

