import os
import shutil
import cv2
import yaml

def prepare_yolo_dataset():
    base_dir = r"c:\Users\meher\Downloads\mini DATASET"
    source_dir = os.path.join(base_dir, "archive", "pothole600")
    dest_dir = os.path.join(base_dir, "yolo_dataset")
    
    # Define splits
    splits = ["training", "validation", "testing"]
    yolo_splits = {
        "training": "train",
        "validation": "val",
        "testing": "test"
    }
    
    # Create directory structure
    for yolo_split in yolo_splits.values():
        os.makedirs(os.path.join(dest_dir, "images", yolo_split), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "labels", yolo_split), exist_ok=True)
        
    print("Converting labels and setting up images...")
    
    for split in splits:
        y_split = yolo_splits[split]
        split_dir = os.path.join(source_dir, split)
        rgb_dir = os.path.join(split_dir, "rgb")
        label_dir = os.path.join(split_dir, "label")
        
        if not os.path.exists(rgb_dir) or not os.path.exists(label_dir):
            print(f"Skipping split '{split}' as rgb or label directory does not exist.")
            continue
            
        files = [f for f in os.listdir(rgb_dir) if f.endswith(".png")]
        print(f"Processing split '{split}' ({len(files)} files) -> '{y_split}'")
        
        for file in files:
            # Copy Image
            img_src = os.path.join(rgb_dir, file)
            img_dest = os.path.join(dest_dir, "images", y_split, file)
            shutil.copy2(img_src, img_dest)
            
            # Read and process label binary mask
            label_src = os.path.join(label_dir, file)
            mask = cv2.imread(label_src, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"  Warning: Cannot read {label_src}, skipping labels for this image.")
                continue
                
            H, W = mask.shape
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            label_txt_file = os.path.splitext(file)[0] + ".txt"
            label_dest = os.path.join(dest_dir, "labels", y_split, label_txt_file)
            
            with open(label_dest, "w") as lf:
                for c in contours:
                    # Ignore extremely small contours (noise)
                    if cv2.contourArea(c) < 4:
                        continue
                        
                    x, y, w, h = cv2.boundingRect(c)
                    
                    # Convert to normalized coordinates
                    x_center = (x + w / 2.0) / W
                    y_center = (y + h / 2.0) / H
                    norm_w = w / W
                    norm_h = h / H
                    
                    # Single class: 0 (pothole)
                    lf.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                    
    # Create dataset.yaml
    yaml_content = {
        "path": dest_dir.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "pothole"
        }
    }
    
    yaml_path = os.path.join(dest_dir, "dataset.yaml")
    with open(yaml_path, "w") as yf:
        yaml.dump(yaml_content, yf, default_flow_style=False)
        
    print(f"YOLO dataset prep complete. dataset.yaml written to: {yaml_path}")

if __name__ == "__main__":
    prepare_yolo_dataset()
