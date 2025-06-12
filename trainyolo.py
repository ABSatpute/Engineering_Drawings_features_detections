from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import matplotlib.pyplot as plt
def main():
    # Training configuration
    model = YOLO("yolo11x-obb.pt")  # Use the OBB model variant (change as needed)
    
    # Start training
    results = model.train(
        data= r'data//engineering_drawings/data.yaml',  # Dataset YAML
        epochs=100,
        imgsz=640,
        batch=4,
        workers=0,
        device='cpu',  # Use 0 for first GPU, 'cpu' for CPU, or '' for auto
        name='obb_engdraw',
        resume=False,
        rect=False,
        exist_ok=False
    )
    print("✅ Training complete.")
    print(results)
 
if __name__ == "__main__":
    main()
