#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
import torch
import traceback
from pathlib import Path

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import our modules
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from load_camera_params import load_camera_params, apply_camera_params_to_estimator

def main():
    """Main function."""
    # Configuration variables (modify these as needed)
    # ===============================================
    
    # Input/Output
    source = 0  # Path to input video file or webcam index (0 for default camera)
    output_path = "output.mp4"  # Path to output video file
    
    # Model settings
    yolo_model_size = "nano"  # YOLOv8 model size: "nano", "small", "medium", "large", "extra"
    depth_model_size = "small"  # Depth Anything v2 model size: "small", "base", "large"
    
    # Device settings
    device = 'cpu'  # Force CPU for stability
    
    # Detection settings
    conf_threshold = 0.25  # Confidence threshold for object detection
    iou_threshold = 0.45  # IoU threshold for NMS
    classes = None  # Filter by class, e.g., [0, 1, 2] for specific classes, None for all classes
    
    # Feature toggles
    enable_tracking = True  # Enable object tracking
    enable_bev = True  # Enable Bird's Eye View visualization
    enable_pseudo_3d = True  # Enable pseudo-3D visualization
    
    # Performance settings
    processing_scale = 0.5  # Scale factor for processing (smaller = faster, but less accurate)
    frame_skip = 0  # Process every N+1 frames (0 = process all frames)
    enable_depth_downsample = True  # Downsample images before depth estimation
    depth_scale = 0.5  # Scale factor for depth estimation (smaller = faster)
    
    # Camera parameters - simplified approach
    camera_params_file = None  # Path to camera parameters file (None to use default parameters)
    # ===============================================
    
    print(f"Using device: {device}")
    
    # Initialize models
    print("Initializing models...")
    try:
        print("Creating object detector instance...")
        # Check if YOLO model file exists
        model_file = f"yolo11{yolo_model_size[0]}.pt"
        if os.path.exists(model_file):
            print(f"YOLO model file found: {model_file}")
        else:
            print(f"WARNING: YOLO model file not found: {model_file}")
            
        detector = ObjectDetector(
            device=device,
            model_size=yolo_model_size,
            verbose=True
        )
        print("Object detector created successfully")
    except Exception as e:
        print(f"Error initializing object detector: {e}")
        print("Falling back to CPU for object detection")
        detector = ObjectDetector(
            device='cpu',
            model_size=yolo_model_size,
            verbose=True
        )
    
    try:
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device=device
        )
    except Exception as e:
        print(f"Error initializing depth estimator: {e}")
        print("Falling back to CPU for depth estimation")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device='cpu'
        )
    
    # Initialize 3D bounding box estimator with default parameters
    # Simplified approach - focus on 2D detection with depth information
    bbox3d_estimator = BBox3DEstimator()
    
    # Initialize Bird's Eye View if enabled
    if enable_bev:
        # Use a scale that works well for the 1-5 meter range
        bev = BirdEyeView(scale=60, size=(300, 300))  # Increased scale to spread objects out
    
    # Open video source
    try:
        if isinstance(source, str) and source.isdigit():
            source = int(source)  # Convert string number to integer for webcam
    except ValueError:
        pass  # Keep as string (for video file)
    
    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:  # Sometimes happens with webcams
        fps = 30
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"
    
    print("Starting processing...")
    
    # Main loop
    frame_idx = 0
    skipped_frames = 0
    
    # Option to skip detections & depth for some frames
    frame_count = 0
    
    # Track results for reuse between frames that we skip
    last_detections = []
    last_depth_map = None
    last_processed_frame = None
    
    try:
        while True:
            if cap.isOpened():
                # Check for key press to exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Only process every N frames based on frame_skip setting
                should_process = True
                if frame_skip > 0:
                    should_process = frame_count % (frame_skip + 1) == 0
                
                if should_process:
                    # Make a copy for different visualizations
                    result_frame = frame.copy()
                    
                    # Resize frame for processing if scale is not 1.0
                    if processing_scale != 1.0:
                        process_width = int(frame.shape[1] * processing_scale)
                        process_height = int(frame.shape[0] * processing_scale)
                        process_frame = cv2.resize(frame, (process_width, process_height))
                    else:
                        process_frame = frame
                    
                    # Object detection
                    try:
                        results = detector.detect(process_frame, conf_threshold=conf_threshold, classes=classes)
                        
                        # Convert YOLO results to our format
                        last_detections = []
                        if results is not None and len(results.boxes) > 0:
                            boxes = results.boxes.xyxy.cpu().numpy()
                            scores = results.boxes.conf.cpu().numpy()
                            class_ids = results.boxes.cls.cpu().numpy()
                            
                            # Check if tracking info is available
                            has_tracking = hasattr(results.boxes, 'id') and results.boxes.id is not None
                            object_ids = results.boxes.id.cpu().numpy() if has_tracking else [None] * len(boxes)
                            
                            for box, score, class_id, obj_id in zip(boxes, scores, class_ids, object_ids):
                                # Rescale boxes if we processed at different resolution
                                if processing_scale != 1.0:
                                    box[0] = box[0] / processing_scale  # xmin
                                    box[1] = box[1] / processing_scale  # ymin
                                    box[2] = box[2] / processing_scale  # xmax
                                    box[3] = box[3] / processing_scale  # ymax
                                
                                last_detections.append([box, score, int(class_id), int(obj_id) if obj_id is not None else None])
                    except Exception as e:
                        print(f"Object detection error: {e}")
                        last_detections = []
                    
                    # Depth estimation
                    try:
                        # Resize for depth estimation if needed
                        if enable_depth_downsample and depth_scale != 1.0:
                            depth_width = int(frame.shape[1] * depth_scale)
                            depth_height = int(frame.shape[0] * depth_scale)
                            depth_frame = cv2.resize(frame, (depth_width, depth_height))
                            
                            # Get depth map
                            last_depth_map = depth_estimator.estimate_depth(depth_frame)
                            
                            # Resize back to original size
                            last_depth_map = cv2.resize(last_depth_map, (frame.shape[1], frame.shape[0]))
                        else:
                            last_depth_map = depth_estimator.estimate_depth(frame)
                    except Exception as e:
                        print(f"Depth estimation error: {e}")
                        # Create dummy depth map
                        last_depth_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
                    
                    last_processed_frame = frame.copy()
                else:
                    # Use the previous results for frames we skip
                    result_frame = frame.copy()
                
                # Process all detections (using cached results if this is a skipped frame)
                active_ids = set()
                
                # 3D bounding box estimation
                for det in last_detections:
                    box, score, class_id, obj_id = det
                    
                    # Estimate depth for this object
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Get class to determine size
                    class_name = detector.get_class_names()[class_id]
                    
                    # Create 3D box based on class
                    if last_depth_map is not None:
                        try:
                            # Calculate center point of object
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Sample depth around center point (more robust)
                            region_size = 5
                            x_start = max(0, center_x - region_size)
                            x_end = min(frame.shape[1] - 1, center_x + region_size)
                            y_start = max(0, center_y - region_size)
                            y_end = min(frame.shape[0] - 1, center_y + region_size)
                            
                            depth_region = last_depth_map[y_start:y_end, x_start:x_end]
                            object_depth = np.median(depth_region)
                            
                            # Track active IDs for cleanup
                            if obj_id is not None:
                                active_ids.add(obj_id)
                        except Exception as e:
                            print(f"Error processing depth for object: {e}")
                            object_depth = 10.0  # Default fallback value
                    else:
                        object_depth = 10.0  # Default fallback value if no depth map
                    
                    # Draw boxes differently based on class
                    if class_id == 0:  # Person
                        color = (0, 255, 0)  # Green
                    elif class_id == 2:  # Car
                        color = (0, 0, 255)  # Red
                    else:
                        color = (255, 0, 0)  # Blue
                    
                    # Draw 2D box
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with depth
                    label = f"{class_name} {score:.2f} D:{object_depth:.1f}m"
                    if obj_id is not None:
                        label = f"ID:{obj_id} " + label
                    
                    cv2.putText(result_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw Bird's Eye View if enabled
                if enable_bev:
                    try:
                        # Reset the BEV image
                        bev_img = np.zeros((bev.height, bev.width, 3), dtype=np.uint8)
                        
                        # Draw objects in BEV
                        for det in last_detections:
                            box, score, class_id, obj_id = det
                            
                            # Get class to determine color
                            if class_id == 0:  # Person
                                color = (0, 255, 0)  # Green
                            elif class_id == 2:  # Car
                                color = (0, 0, 255)  # Red
                            else:
                                color = (255, 0, 0)  # Blue
                            
                            # Calculate position in BEV
                            x1, y1, x2, y2 = box.astype(int)
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Sample depth at center
                            if last_depth_map is not None:
                                try:
                                    # Sample depth around center (more robust)
                                    region_size = 5
                                    x_start = max(0, center_x - region_size)
                                    x_end = min(frame.shape[1] - 1, center_x + region_size)
                                    y_start = max(0, center_y - region_size)
                                    y_end = min(frame.shape[0] - 1, center_y + region_size)
                                    
                                    depth_region = last_depth_map[y_start:y_end, x_start:x_end]
                                    z = np.median(depth_region)
                                    
                                    # Transform to BEV coordinates
                                    bev_x = int(bev.width - (z * bev.scale))
                                    # Normalize x position to be relative to frame width
                                    rel_x = center_x / frame.shape[1]
                                    # Map to BEV y-coordinate (left-right in BEV)
                                    bev_y = int(rel_x * bev.height)
                                    
                                    # Ensure within bounds
                                    bev_x = max(0, min(bev.width - 1, bev_x))
                                    bev_y = max(0, min(bev.height - 1, bev_y))
                                    
                                    # Draw object in BEV
                                    size = 5 if class_id == 0 else 8  # Person vs Car size
                                    cv2.circle(bev_img, (bev_x, bev_y), size, color, -1)
                                    
                                    # Add ID if available
                                    if obj_id is not None:
                                        cv2.putText(bev_img, f"{obj_id}", (bev_x - 10, bev_y - 10), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                except Exception as e:
                                    pass  # Skip drawing this object in BEV
                        
                        # Make sure BEV dimensions are valid before resizing
                        if bev_img.shape[0] > 0 and bev_img.shape[1] > 0:
                            # Resize for display (fits in corner of frame)
                            bev_display_width = int(frame.shape[1] * 0.3)
                            bev_display_height = int(bev.height * (bev_display_width / bev.width))
                            bev_display = cv2.resize(bev_img, (bev_display_width, bev_display_height))
                            
                            # Create ROI in the result frame
                            roi_y = 20
                            roi_x = 20
                            
                            # Make sure we don't try to create ROI outside image boundaries
                            if roi_y + bev_display_height > result_frame.shape[0]:
                                roi_y = result_frame.shape[0] - bev_display_height - 5
                            
                            if roi_x + bev_display_width > result_frame.shape[1]:
                                roi_x = result_frame.shape[1] - bev_display_width - 5
                            
                            # Ensure coordinates are valid
                            roi_y = max(0, roi_y)
                            roi_x = max(0, roi_x)
                            
                            roi = result_frame[roi_y:roi_y+bev_display_height, roi_x:roi_x+bev_display_width]
                            
                            # Only proceed if ROI shape matches BEV display dimensions
                            if roi.shape[:2] == bev_display.shape[:2]:
                                # Create a mask for transparent overlay
                                mask = np.any(bev_display > 0, axis=2).astype(np.uint8) * 255
                                mask_inv = cv2.bitwise_not(mask)
                                
                                # Resize masks if needed to match ROI
                                if mask.shape[:2] != roi.shape[:2]:
                                    mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
                                    mask_inv = cv2.resize(mask_inv, (roi.shape[1], roi.shape[0]))
                                
                                # Convert masks to 3 channels
                                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
                                mask_inv_3ch = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR) / 255.0
                                
                                # Overlay the BEV on the ROI
                                bev_part = cv2.multiply(bev_display.astype(float), mask_3ch)
                                roi_part = cv2.multiply(roi.astype(float), mask_inv_3ch)
                                roi_result = cv2.add(bev_part, roi_part).astype(np.uint8)
                                
                                # Update the ROI in the result frame
                                result_frame[roi_y:roi_y+bev_display_height, roi_x:roi_x+bev_display_width] = roi_result
                                
                                # Draw border around BEV
                                cv2.rectangle(result_frame, (roi_x, roi_y), 
                                             (roi_x+bev_display_width, roi_y+bev_display_height), 
                                             (255, 255, 255), 2)
                                
                                # Add title above the BEV
                                cv2.putText(result_frame, "Bird's Eye View", (roi_x, max(5, roi_y-5)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    except Exception as e:
                        print(f"Error drawing BEV: {e}")
                
                # Calculate and display fps (update every 10 frames)
                frame_time = time.time() - start_time
                start_time = time.time()
                
                # Update FPS with moving average
                fps_value = 1.0 / frame_time if frame_time > 0 else 0
                fps_display = f"FPS: {fps_value:.1f}"
                
                # Add FPS and device to frame
                cv2.putText(result_frame, f"{fps_display} | Device: {device}", 
                           (10, result_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 1)
                
                # Add skip info if frame skipping is enabled
                if frame_skip > 1:
                    cv2.putText(result_frame, f"Processing: 1/{frame_skip} frames", 
                               (result_frame.shape[1] - 200, result_frame.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Add depth map corner
                if enable_depth_downsample and last_depth_map is not None:
                    try:
                        # Normalize depth map for visualization
                        depth_vis = cv2.normalize(last_depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                        
                        # Resize depth map for corner display
                        depth_corner_height = int(result_frame.shape[0] * 0.25)
                        depth_corner_width = int(result_frame.shape[1] * 0.25)
                        depth_corner = cv2.resize(depth_color, (depth_corner_width, depth_corner_height))
                        
                        # Position in bottom-right corner
                        x_offset = result_frame.shape[1] - depth_corner_width - 20
                        y_offset = result_frame.shape[0] - depth_corner_height - 20
                        
                        # Create ROI in the result frame
                        roi = result_frame[y_offset:y_offset+depth_corner_height, x_offset:x_offset+depth_corner_width]
                        
                        # Add semi-transparent overlay
                        alpha = 0.7
                        cv2.addWeighted(depth_corner, alpha, roi, 1-alpha, 0, roi)
                        
                        # Update the ROI in the result frame
                        result_frame[y_offset:y_offset+depth_corner_height, x_offset:x_offset+depth_corner_width] = roi
                        
                        # Add border around depth map
                        cv2.rectangle(result_frame, (x_offset, y_offset), 
                                     (x_offset+depth_corner_width, y_offset+depth_corner_height), 
                                     (255, 255, 255), 1)
                        
                        # Add title
                        cv2.putText(result_frame, "Depth Map", (x_offset, y_offset-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    except Exception as e:
                        print(f"Error adding depth map: {e}")
                
                # Write to video
                out.write(result_frame)
                
                # Display the frame
                cv2.imshow('3D Object Detection', result_frame)
                
                # Add slight delay to reduce CPU usage
                time.sleep(0.001)
            else:
                break
    except Exception as e:
        print(f"Error processing frames: {e}")
        print(f"Detailed error: {traceback.format_exc()}")
    
    # Clean up
    print("Cleaning up resources...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        # Clean up OpenCV windows
        cv2.destroyAllWindows() 