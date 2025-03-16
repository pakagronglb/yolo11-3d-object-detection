#!/usr/bin/env python3
import os
import sys
import traceback
import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)

# Load environment variables (for Hugging Face token)
load_dotenv()
print(f"HF_TOKEN environment variable {'is set' if os.environ.get('HF_TOKEN') else 'is NOT set'}")

print("\n=== Testing DETECTION MODEL ===")
try:
    print("Checking for YOLO model file...")
    model_file = "yolo11n.pt"
    if os.path.exists(model_file):
        print(f"✓ YOLO model file found: {model_file} ({os.path.getsize(model_file) / (1024*1024):.1f} MB)")
    else:
        print(f"✗ YOLO model file NOT found: {model_file}")
        sys.exit(1)
        
    print("\nImporting ultralytics...")
    from ultralytics import YOLO
    print("✓ Ultralytics imported successfully")
    
    print("\nLoading YOLO model...")
    model = YOLO(model_file)
    print(f"✓ YOLO model loaded successfully: {model}")
    print(f"Model task: {model.task}")
    print(f"Model names: {model.names}")
    
    # Test inference on a blank image
    print("\nTesting inference...")
    blank_image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(blank_image, verbose=True)
    print(f"✓ Inference successful: {results}")
    
except Exception as e:
    print(f"\n✗ DETECTION MODEL ERROR: {e}")
    print("\nDetailed error traceback:")
    traceback.print_exc()
    
print("\n\n=== Testing DEPTH MODEL ===")
try:
    print("Importing transformers...")
    from transformers import pipeline
    print("✓ Transformers imported successfully")
    
    print("\nInitializing depth estimation pipeline...")
    pipe_device = 'cpu'
    model_name = 'depth-anything/Depth-Anything-V2-Small-hf'
    pipe = pipeline(task="depth-estimation", model=model_name, device=pipe_device)
    print(f"✓ Depth model loaded successfully")
    
    # Test inference on a blank image
    print("\nTesting inference...")
    from PIL import Image
    blank_image = Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
    depth_map = pipe(blank_image)
    print(f"✓ Inference successful: {depth_map}")
    
except Exception as e:
    print(f"\n✗ DEPTH MODEL ERROR: {e}")
    print("\nDetailed error traceback:")
    traceback.print_exc()

print("\n=== Debug Complete ===") 