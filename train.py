from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import matplotlib.pyplot as plt
def main():
    # Training configuration
    model = YOLO("yolo11x-obb.pt")  # Use the OBB model variant (change as needed)

    train_args = {
        'data': r'C://Users//DELL//Desktop//YOLOv11-OBB//data//engineering_drawings/data.yaml',
        'epochs': 100,
        'imgsz': 640,
        'device': 'cuda'*,
        'batch': 16,
        'name': 'engineering_drawings_yolo11m',
        'project': 'runs/detect',
        'save_period': 10,
        'val': True,
        'plots': True,
        'verbose': True,

        # Data Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.9,
        'shear': 2.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.3,

        # Training hyperparameters
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'patience': 50,
        'close_mosaic': 10,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
    }

    # # Start training
    # results = model.train(
    #     data= r'data//engineering_drawings/data.yaml',  # Dataset YAML
    #     epochs=100,
    #     imgsz=640,
    #     batch=4,
    #     workers=0,
    #     device='cpu',  # Use 0 for first GPU, 'cpu' for CPU, or '' for auto
    #     name='obb_engdraw',
    #     resume=False,
    #     rect=False,
    #     exist_ok=False
    # )

    results = model.train(**train_args)
    print("✅ Training completed successfully!")
    print(results)
 
if __name__ == "__main__":
    main()
