"""
Train YOLOv8n on Substation Asset Dataset.
Runs on local NVIDIA RTX 4060 GPU with CUDA.
"""

import os
import shutil
from ultralytics import YOLO

def train():
    yaml_path = os.path.abspath(os.path.join("yolo_dataset", "dataset.yaml"))
    print(f"[TRAIN] Loading base model: yolov8n.pt...")
    model = YOLO("yolov8n.pt")

    print("[TRAIN] Starting training on GPU...")
    results = model.train(
        data=yaml_path,
        epochs=60,
        imgsz=640,
        batch=8,
        device=0,  # GPU 0
        name="tcs_asset_model",
        project="runs/detect",
        exist_ok=True,
        verbose=True,
        patience=20,
        save=True,
    )
    print("\n[TRAIN] Training complete!")
    best_weights = os.path.abspath(os.path.join(results.save_dir, "weights", "best.pt"))
    print(f"[TRAIN] Best weights saved at: {best_weights}")

    # Copy to weights/best.pt for production inference
    target_weights = os.path.abspath(os.path.join("weights", "best.pt"))
    os.makedirs(os.path.dirname(target_weights), exist_ok=True)
    if os.path.exists(best_weights):
        shutil.copy2(best_weights, target_weights)
        print(f"[TRAIN] Successfully copied best weights to: {target_weights}")

    return target_weights

if __name__ == "__main__":
    train()
