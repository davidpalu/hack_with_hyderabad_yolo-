#!/usr/bin/env python3


import argparse
import os
from ultralytics import YOLO
import torch


EPOCHS = 12              
MOSAIC = 0.4
OPTIMIZER = "AdamW"        
MOMENTUM = 0.937            
LR0 = 0.002                 
LRF = 0.01                  
WEIGHT_DECAY = 0.0005
PATIENCE = 30              
IMG_SIZE = 640
BATCH = 8                  
SINGLE_CLS = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--mosaic", type=float, default=MOSAIC, help="Mosaic augmentation")
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="Optimizer")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum (SGD only)")
    parser.add_argument("--lr0", type=float, default=LR0, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=LRF, help="Final LR multiplier")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Image size")
    parser.add_argument("--batch", type=int, default=BATCH, help="Batch size (int)")
    parser.add_argument("--single_cls", type=bool, default=SINGLE_CLS, help="Single class training")

    args = parser.parse_args()

    # Set working directory
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    # Pick device automatically (prefer CUDA if available)
    device = "0" if torch.cuda.is_available() else "cpu"

    # Load YOLOv8s model
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))

    # Build training args
    train_args = {
        "data": os.path.join(this_dir, "yolo_params.yaml"),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "single_cls": args.single_cls,
        "mosaic": args.mosaic,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "verbose": True,
    }

    # Momentum only for SGD
    if args.optimizer.lower() == "sgd":
        train_args["momentum"] = args.momentum

    # Train
    print(f"[INFO] Training {args.optimizer} for {args.epochs} epochs on {device}")
    results = model.train(**train_args)

    # Validate after training
    print("[INFO] Validating best checkpoint...")
    model.val(data=os.path.join(this_dir, "yolo_params.yaml"), split="val", device=device)

    print("[DONE] Training complete. Check runs/detect/trainX for results.")


'''
Mixup boost val pred but reduces test pred
Mosaic shouldn't be 1.0  
'''


'''
                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
'''
