import os
import cv2
import numpy as np
from ultralytics import YOLO

def scan():
    test_dir = r"c:\Users\meher\Downloads\mini DATASET\archive\pothole600\testing\rgb"
    model_path = r"c:\Users\meher\Downloads\mini DATASET\models\best.pt"
    model = YOLO(model_path)
    
    results = []
    for f in sorted(os.listdir(test_dir)):
        p = os.path.join(test_dir, f)
        img = cv2.imread(p)
        if img is None:
            continue
        res = model(img, conf=0.15, verbose=False)
        count = len(res[0].boxes)
        results.append((f, count))
        
    results.sort(key=lambda x: x[1], reverse=True)
    print("Top 10 images with highest pothole detections:")
    for f, c in results[:10]:
        print(f"File: {f}, Detection Count: {c}")

if __name__ == "__main__":
    scan()
