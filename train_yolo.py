import os
from ultralytics import YOLO

def train():
    base_dir = r"c:\Users\meher\Downloads\mini DATASET"
    yaml_path = os.path.join(base_dir, "yolo_dataset", "dataset.yaml")
    
    print("Initializing YOLOv8-Nano model...")
    # Load a pre-trained yolov8n model
    model = YOLO("yolov8n.pt")
    
    print("Starting training on GPU (CUDA)...")
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=15,
        imgsz=400,
        device=0,         # Use Nvidia GPU
        workers=2,
        project=os.path.join(base_dir, "runs"),
        name="pothole_detection",
        exist_ok=True
    )
    
    print("Training finished!")
    
    # Copy best weights to models directory
    best_weights_src = os.path.join(base_dir, "runs", "pothole_detection", "weights", "best.pt")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_weights_dest = os.path.join(models_dir, "best.pt")
    
    if os.path.exists(best_weights_src):
        import shutil
        shutil.copy2(best_weights_src, best_weights_dest)
        print(f"Best model weights saved to: {best_weights_dest}")
    else:
        print("Warning: could not find training weights at:", best_weights_src)

if __name__ == "__main__":
    train()
