# 🚧 Road Damage Detection & Severity Estimation

An intelligent municipal road inspection system using **YOLOv8-Nano** for real-time pothole detection, severity classification, and priority-based road health scoring — with an interactive Streamlit dashboard.

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🔍 **Pothole Detection** | YOLOv8-Nano trained on the Pothole600 dataset |
| 📊 **Severity Estimation** | Mild / Moderate / Severe based on bounding-box area and LiDAR disparity depth |
| 🚦 **Priority Scoring** | Critical / High / Medium / Low road health priority with recommended action |
| 📸 **Live Photo Mode** | Upload any road photo or snap one via webcam for instant analysis |
| 🗺️ **GPS Mapping** | Interactive Folium map with pothole markers color-coded by severity |
| ⬇️ **Export** | Download annotated detection images |

---

## 🚦 Road Health Score

The system calculates a **Road Health Score (0–100)** based on detected damage:

- 🟡 **Mild** pothole → −5 points  
- 🟠 **Moderate** pothole → −10 points  
- 🔴 **Severe** pothole → −20 points  

### Priority Classification

| Score | Severe Count | Priority | Action |
|-------|-------------|----------|--------|
| < 40 | ≥ 3 | 🔴 Critical | Immediate repair — road unsafe |
| < 60 | ≥ 1 | 🟠 High | Repair within 7 days |
| < 80 | — | 🟡 Medium | Maintenance within 30 days |
| ≥ 80 | 0 | 🟢 Low | Routine monitoring |

---

## 📂 Project Structure

```
road-damage-detection/
├── dashboard.py          # Streamlit dashboard (main app)
├── severity_estimator.py # Pothole metrics & road health scoring
├── train_yolo.py         # YOLOv8 training script
├── prepare_dataset.py    # Dataset preparation
├── inspect_data.py       # Data inspection utilities
├── models/               # Trained weights (best.pt — not tracked in git)
├── tests/
│   └── test_pipeline.py  # Unit tests
└── requirements.txt      # Python dependencies
```

---

## 🛠️ Installation

```bash
git clone https://github.com/reshwanth9912/road-damage-detection.git
cd road-damage-detection
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the model (optional — use provided weights)
```bash
python train_yolo.py
```

### 2. Launch the dashboard
```bash
streamlit run dashboard.py
```

Access at **http://localhost:8501**

---

## 📸 Dashboard Modes

### 📂 Dataset Browse
Use the sidebar slider to browse through pre-indexed frames from the Pothole600 test set.  
Requires the dataset files in `archive/pothole600/testing/`.

### 📸 Live Photo / Upload
- **Upload Tab**: Drag & drop or browse to upload any JPG/PNG road photo.
- **Webcam Tab**: Click "Take Photo" to capture a live road image from your camera.
- Results include bounding boxes, severity labels, metric cards, priority alert, damage table, and GPS map.

---

## 📦 Requirements

```
ultralytics>=8.0.0
streamlit>=1.28.0
opencv-python-headless
numpy
pillow
folium
pandas
```

---

## 📊 Dataset

Built on the [Pothole-600 Dataset](https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset) — 600 RGB images paired with stereo disparity maps for depth estimation.

---

## 🤝 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [Folium](https://python-visualization.github.io/folium/)
