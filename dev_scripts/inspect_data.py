import os
from PIL import Image
import numpy as np

def inspect():
    base_dir = r"c:\Users\meher\Downloads\mini DATASET"
    train_dir = os.path.join(base_dir, "archive", "pothole600", "training")
    
    rgb_path = os.path.join(train_dir, "rgb", "0000.png")
    label_path = os.path.join(train_dir, "label", "0000.png")
    tdisp_path = os.path.join(train_dir, "tdisp", "0000.png")
    
    print("Checking file presence:")
    print(f"RGB exists: {os.path.exists(rgb_path)}")
    print(f"Label exists: {os.path.exists(label_path)}")
    print(f"Tdisp exists: {os.path.exists(tdisp_path)}")
    
    if os.path.exists(rgb_path):
        rgb_img = Image.open(rgb_path)
        print(f"\nRGB Image Info:")
        print(f"  Format: {rgb_img.format}")
        print(f"  Size (W, H): {rgb_img.size}")
        print(f"  Mode: {rgb_img.mode}")
        rgb_arr = np.array(rgb_img)
        print(f"  Numpy Shape: {rgb_arr.shape}")
        print(f"  Type: {rgb_arr.dtype}")
        print(f"  Range: [{rgb_arr.min()}, {rgb_arr.max()}]")
        
    if os.path.exists(label_path):
        label_img = Image.open(label_path)
        print(f"\nLabel Image Info:")
        print(f"  Format: {label_img.format}")
        print(f"  Size (W, H): {label_img.size}")
        print(f"  Mode: {label_img.mode}")
        label_arr = np.array(label_img)
        print(f"  Numpy Shape: {label_arr.shape}")
        print(f"  Type: {label_arr.dtype}")
        print(f"  Unique values: {np.unique(label_arr)}")
        
    if os.path.exists(tdisp_path):
        tdisp_img = Image.open(tdisp_path)
        print(f"\nTdisp Image Info:")
        print(f"  Format: {tdisp_img.format}")
        print(f"  Size (W, H): {tdisp_img.size}")
        print(f"  Mode: {tdisp_img.mode}")
        tdisp_arr = np.array(tdisp_img)
        print(f"  Numpy Shape: {tdisp_arr.shape}")
        print(f"  Type: {tdisp_arr.dtype}")
        print(f"  Range: [{tdisp_arr.min()}, {tdisp_arr.max()}]")
        print(f"  Unique values / distribution (truncated check): {np.unique(tdisp_arr)[:20] if len(np.unique(tdisp_arr)) > 20 else np.unique(tdisp_arr)}")

if __name__ == "__main__":
    inspect()
