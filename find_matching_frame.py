import os
import cv2
import numpy as np

def find_match():
    test_dir = r"c:\Users\meher\Downloads\mini DATASET\archive\pothole600\testing\rgb"
    screenshot_path = r"C:/Users/meher/.gemini/antigravity/brain/f63e614f-63b3-4e5d-8973-253f15e35f19/uploaded_media_1783703097902.png"
    
    screen = cv2.imread(screenshot_path)
    if screen is None:
        print("Screenshot not found at:", screenshot_path)
        return
        
    h_s, w_s, _ = screen.shape
    # Left crop of the dashboard screenshot contains the original image
    left_img = screen[:, :int(w_s*0.48)]
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    
    best_val = -1
    best_file = ""
    
    for f in sorted(os.listdir(test_dir)):
        p = os.path.join(test_dir, f)
        if not os.path.exists(p):
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to match template scale
        resized_img = cv2.resize(img_gray, (left_gray.shape[1], left_gray.shape[0]))
        res = cv2.matchTemplate(left_gray, resized_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > best_val:
            best_val = max_val
            best_file = f
            
    print(f"Best match image: {best_file} (Correlation: {best_val:.4f})")

if __name__ == "__main__":
    find_match()
