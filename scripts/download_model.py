"""
download_model.py — Pre-download YOLOv8n pretrained weights

Run this ONCE before launching the app:
    python download_model.py

The model file (yolov8n.pt ~6 MB) will be saved in this folder.
Ultralytics downloads it from:  https://github.com/ultralytics/assets
"""

import os
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "yolov8n.pt"          # Nano model — fastest, ~6 MB
MODEL_PATH = BASE_DIR / "weights" / MODEL_NAME


def download_yolo_model():
    print("=" * 55)
    print("  YOLOv8 Pretrained Model Downloader")
    print("=" * 55)

    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"\n✅  Model already exists: {MODEL_PATH}")
        print(f"    Size: {size_mb:.2f} MB")
        print("\nNo download needed. You're good to go!")
    else:
        print(f"\n📥  Downloading:  {MODEL_NAME}")
        print(f"    Destination: {MODEL_PATH}")
        print(f"    Source:      https://github.com/ultralytics/assets\n")

        # YOLO() auto-downloads the weights if we pass "yolov8n.pt", 
        # but to save it exactly where we want it without weird caching logic:
        model = YOLO(MODEL_NAME)
        # Move it to weights if it downloaded to root
        root_model = BASE_DIR / MODEL_NAME
        if root_model.exists():
            root_model.rename(MODEL_PATH)

        if MODEL_PATH.exists():
            size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            print(f"\n✅  Download complete!")
            print(f"    Saved to:  {MODEL_PATH}")
            print(f"    Size:      {size_mb:.2f} MB")
        else:
            print(f"\n⚠  Model saved to default Ultralytics cache (not local folder).")
            print(f"   This is fine — the app will load it from cache automatically.")

    print("\n" + "=" * 55)
    print("  Classes this model can detect (80 COCO classes):")
    print("=" * 55)

    # Print all detectable class names
    model = YOLO(MODEL_NAME)
    names = model.names           # dict: {id: "label"}
    for i in range(0, len(names), 4):
        row = [f"{i+j:>3}. {names[i+j]:<18}"
               for j in range(4) if (i + j) < len(names)]
        print("  " + "".join(row))

    print("\n✅  Ready! Run the app with:  streamlit run src/streamlit_app.py")
    print("=" * 55)

if __name__ == "__main__":
    download_yolo_model()
