import sys, os
from PIL import Image
import numpy as np
from ultralytics import YOLO

def test_adaptive():
    model_path = r"c:\Users\meher\Downloads\mini DATASET\models\best.pt"
    model = YOLO(model_path)
    
    img0_path = r"C:/Users/meher/.gemini/antigravity/brain/f63e614f-63b3-4e5d-8973-253f15e35f19/uploaded_media_0_1783703435815.png"
    img1_path = r"C:/Users/meher/.gemini/antigravity/brain/f63e614f-63b3-4e5d-8973-253f15e35f19/uploaded_media_1783703097902.png"
    
    for label, path in [("Single Pothole", img0_path), ("Multiple Potholes", img1_path)]:
        img = Image.open(path)
        arr = np.array(img.convert("RGB"))
        
        # 1. Run inference at conf=0.01 to see candidates
        results = model(arr, conf=0.01, iou=0.3, imgsz=640, verbose=False)
        boxes = results[0].boxes
        confs = [float(b.conf[0]) for b in boxes]
        
        # 2. Compute dynamic threshold
        if len(confs) > 0:
            max_conf = max(confs)
            if len(confs) >= 8:
                dynamic_conf = 0.015
            elif max_conf > 0.60:
                dynamic_conf = 0.12
            else:
                dynamic_conf = 0.05
        else:
            dynamic_conf = 0.15
            
        # 3. Filter final list
        final_boxes = []
        for box in boxes:
            conf = float(box.conf[0])
            if conf >= dynamic_conf:
                coords = box.xyxy[0].cpu().numpy()
                final_boxes.append((coords, conf))
                
        print(f"\n{label}:")
        print(f"  Candidates count at 0.01: {len(confs)}")
        print(f"  Max confidence = {max_conf:.1%}" if confs else "  No candidates")
        print(f"  Chosen dynamic conf: {dynamic_conf:.3f}")
        print(f"  Final detection count: {len(final_boxes)}")
        for idx, (coords, conf) in enumerate(final_boxes):
            print(f"    #{idx+1}: conf={conf:.1%} box={coords.tolist()}")

if __name__ == "__main__":
    test_adaptive()
