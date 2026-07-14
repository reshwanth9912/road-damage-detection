"""
Batch pothole detection across all test images.
Reports per-image counts, severity breakdown, and overall statistics.
"""
import os
import sys
import numpy as np
from PIL import Image
from ultralytics import YOLO
from severity_estimator import estimate_pothole_metrics, calculate_road_health_score

# ── Config ──
BASE_DIR = r"c:\Users\meher\Downloads\mini DATASET"
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
TEST_RGB_DIR = os.path.join(BASE_DIR, "archive", "pothole600", "testing", "rgb")
TEST_DISP_DIR = os.path.join(BASE_DIR, "archive", "pothole600", "testing", "tdisp")

# ── Load Model ──
print("Loading YOLO11 model...")
model = YOLO(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}\n")

# ── Process All Test Images ──
test_files = sorted([f for f in os.listdir(TEST_RGB_DIR) if f.lower().endswith(".png")])
print(f"Total test images: {len(test_files)}")
print("=" * 90)
print(f"{'Image':<14} {'Potholes':>9} {'Severe':>8} {'Moderate':>10} {'Mild':>6} {'Health':>8} {'Confidences'}")
print("-" * 90)

total_potholes = 0
total_severe = 0
total_moderate = 0
total_mild = 0
images_with_detections = 0
all_confidences = []
per_image_results = []

for fname in test_files:
    rgb_path = os.path.join(TEST_RGB_DIR, fname)
    disp_path = os.path.join(TEST_DISP_DIR, fname)

    rgb_pil = Image.open(rgb_path)
    img_arr = np.array(rgb_pil.convert("RGB"))
    H, W = img_arr.shape[:2]

    # Load disparity if available
    if os.path.exists(disp_path):
        disp_arr = np.array(Image.open(disp_path).convert("L"))
    else:
        disp_arr = np.full((H, W), 128, dtype=np.uint8)

    # Run detection
    results = model(img_arr, verbose=False)
    boxes = results[0].boxes

    severities = []
    confs = []
    for box in boxes:
        coords = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        confs.append(conf)
        all_confidences.append(conf)

        metrics = estimate_pothole_metrics(
            [coords[0], coords[1], coords[2], coords[3]], disp_arr
        )
        severities.append(metrics["severity"])

    count = len(boxes)
    s_count = severities.count("Severe")
    m_count = severities.count("Moderate")
    l_count = severities.count("Mild")
    health = calculate_road_health_score(severities)

    total_potholes += count
    total_severe += s_count
    total_moderate += m_count
    total_mild += l_count
    if count > 0:
        images_with_detections += 1

    conf_str = ", ".join(f"{c:.0%}" for c in confs) if confs else "—"
    print(f"{fname:<14} {count:>9} {s_count:>8} {m_count:>10} {l_count:>6} {health:>7}/100  {conf_str}")

    per_image_results.append({
        "file": fname, "count": count,
        "severe": s_count, "moderate": m_count, "mild": l_count,
        "health": health
    })

# ── Summary ──
print("=" * 90)
print(f"\n📊 DETECTION SUMMARY")
print(f"   Total test images:          {len(test_files)}")
print(f"   Images with detections:     {images_with_detections} ({images_with_detections/len(test_files)*100:.1f}%)")
print(f"   Images with NO detections:  {len(test_files) - images_with_detections}")
print(f"\n   Total potholes detected:    {total_potholes}")
print(f"     🔴 Severe:   {total_severe}")
print(f"     🟠 Moderate: {total_moderate}")
print(f"     🟡 Mild:     {total_mild}")
print(f"\n   Avg potholes per image:     {total_potholes/len(test_files):.2f}")
if all_confidences:
    print(f"   Avg confidence:             {np.mean(all_confidences):.1%}")
    print(f"   Min confidence:             {np.min(all_confidences):.1%}")
    print(f"   Max confidence:             {np.max(all_confidences):.1%}")

# Top 5 most damaged
sorted_imgs = sorted(per_image_results, key=lambda x: x["count"], reverse=True)
print(f"\n   🏚️ Top 5 Most Damaged Images:")
for r in sorted_imgs[:5]:
    print(f"     {r['file']}: {r['count']} potholes (Health: {r['health']}/100) — 🔴{r['severe']} 🟠{r['moderate']} 🟡{r['mild']}")

avg_health = np.mean([r["health"] for r in per_image_results])
print(f"\n   🏥 Average Road Health Score: {avg_health:.1f}/100")
print()
