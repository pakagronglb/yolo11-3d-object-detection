import os
import torch
import numpy as np
import cv2
import traceback
from ultralytics import YOLO
from collections import deque
from pathlib import Path

# Check NumPy version and warn if using 2.x
if np.__version__.startswith('2.'):
    print("WARNING: Using NumPy 2.x which may cause compatibility issues with some modules.")
    print("Consider downgrading to NumPy 1.x if you encounter errors.")

class ObjectDetector:
    """
    Object detection using YOLO from Ultralytics
    """
    def __init__(self, device="cpu", model_size="medium", cache_models=True, verbose=False):
        """
        Initialize the YOLO detection model from the ultralytics package.
        
        Args:
            device (str): Device to run the model on ('cpu', 'cuda', 'mps')
            model_size (str): Size of the model ('nano', 'small', 'medium', 'large', 'extra_large')
            cache_models (bool): Whether to cache the model (for faster loading if used multiple times)
        """
        self.device = device
        self.verbose = verbose  # Add verbose flag to control printing
        
        # Check NumPy version
        if np.__version__.startswith('2.'):
            print("WARNING: NumPy 2.x detected. This may cause compatibility issues with YOLO models.")
        
        # Map model size to the corresponding YOLOv8 model file name
        model_map = {
            "nano": "yolov8n",
            "small": "yolov8s",
            "medium": "yolov8m",
            "large": "yolov8l",
            "extra_large": "yolov8x",
        }
        
        # Get the model file name based on the specified size (default to medium if size not found)
        self.model_file = model_map.get(model_size.lower(), "yolov8m")
        
        # Flag to track if fallback model was used
        self.using_fallback = False
        
        # Try to load the model, with fallback options
        try:
            if self.verbose:
                print(f"Loading {self.model_file} on {device}...")
            
            self.model = YOLO(self.model_file)
            
            if self.verbose:
                print(f"Model {self.model_file} loaded successfully!")
        except Exception as e:
            print(f"Error loading model {self.model_file}: {e}")
            try:
                # Try fallback to yolov8n (smallest model)
                print("Attempting to load fallback model (yolov8n)...")
                self.model_file = "yolov8n"
                self.using_fallback = True
                
                self.model = YOLO(self.model_file)
                
                print("Fallback model loaded successfully!")
            except Exception as e2:
                raise RuntimeError(f"Failed to load model and fallback model: {e2}")
        
        # Set model parameters
        self.model.overrides['conf'] = 0.25
        self.model.overrides['iou'] = 0.45
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        
        # Initialize tracking trajectories
        self.tracking_trajectories = {}
    
    def detect(self, image, conf_threshold=0.25, classes=None):
        """
        Detect objects in an image.
        
        Args:
            image: Input image (numpy array)
            conf_threshold: Confidence threshold for detections
            classes: List of classes to detect (None for all classes)
            
        Returns:
            results: YOLO results object
        """
        try:
            # Run inference
            results = self.model(image, conf=conf_threshold, classes=classes)
            
            # Log model info once in a while
            if hasattr(self, 'detection_count'):
                self.detection_count += 1
            else:
                self.detection_count = 1
                
            if self.detection_count % 100 == 1 and self.verbose:
                print(f"Using model: {self.model_file} on {self.device}")
                if self.using_fallback:
                    print("NOTE: Using fallback model due to loading issues with primary model.")
            
            return results[0]  # Return the first (and only) result
        except Exception as e:
            print(f"Error during detection: {e}")
            # Return empty result
            return None
    
    def get_class_names(self):
        """
        Get the names of the classes that the model can detect
        
        Returns:
            list: List of class names
        """
        return self.model.names 