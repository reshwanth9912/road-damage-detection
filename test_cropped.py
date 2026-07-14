import sys, os
from PIL import Image
import numpy as np
from ultralytics import YOLO

def test_cropped():
    model_path = r"c:\Users\meher\Downloads\mini DATASET\models\best.pt"
    model = YOLO(model_path)
    
    img0_path = r"C:/Users/meher/.gemini/antigravity/brain/f63e614f-63b3-4e5d-8973-253f15e35f19/uploaded_media_0_1783703435815.png"
    img1_path = r"C:/Users/meher/.gemini/antigravity/brain/f63e614f-63b3-4e5d-8973-253f15e35f19/uploaded_media_1783703097902.png"
    
    for label, path in [("Raw Single Pothole", img0_path), ("Raw Multiple Potholes", img1_path)]:
        img = Image.open(path)
        w, h = img.size
        # Crop the left half (representing the uploaded road photo)
        # Exclude the dashboard margins by cropping w*0.05 to w*0.47
        cropped_img = img.crop((int(w * 0.057), int(h * 0.18), int(w * 0.468), int(h * 0.94)))
        arr = np.array(cropped_img.convert("RGB"))
        
        # Save cropped for verification
        cropped_img.save(f"cropped_{label.lower().replace(' ', '_')}.png")
        
        # Test detection at low confidence
        results = model(arr, conf=0.01, iou=0.3, imgsz=640, verbose=False)
        boxes = results[0].boxes
        confs = [float(b.conf[0]) for b in boxes]
        
        print(f"\n{label} ({cropped_img.size}):")
        print(f"  Total detections at 0.01: {len(confs)}")
        if confs:
            print(f"  Max conf: {max(confs):.1%}")
            print(f"  Min conf: {min(confs):.1%}")
            print(f"  Confs: {', '.join(f'{c:.1%}' for c in sorted(confs, reverse=True))}")

if __name__ == "__main__":
    test_cropped()
