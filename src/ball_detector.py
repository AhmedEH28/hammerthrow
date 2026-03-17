import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import os
import shutil
import pandas as pd
from scipy.interpolate import CubicSpline
import subprocess

class BallTracker:
    def __init__(self, buffer_size=10, max_distance_threshold=0.15):
        self.buffer_size = buffer_size
        self.buffer = []
        self.max_distance_threshold = max_distance_threshold
        self.init_kalman_filter()
        self.trajectory_path = []
        self.max_trajectory_points = 6

    def init_kalman_filter(self):
        # Standard Kalman filter with minimal prediction influence
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, dx, dy]
        self.kf.F = np.array([[1, 0, 0.5, 0],     # Reduced velocity influence
                             [0, 1, 0, 0.5],
                             [0, 0, 0.8, 0],      # Dampen velocity persistence
                             [0, 0, 0, 0.8]])
        self.kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])
        self.kf.R *= 0.03      # Balance between measurement and prediction
        self.kf.Q *= 0.01      # Low process noise
        self.kf.P *= 5         # Initial uncertainty
        self.kf_initialized = False  # Flag to track initialization status

    def validate_detection(self, detection, frame_width, frame_height):
        x_center, y_center = detection
        
        # Add to buffer
        self.buffer.append((x_center, y_center))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        # Not enough detections yet
        if len(self.buffer) < 3:
            return True
        
        # Calculate distances between consecutive points
        distances = []
        for i in range(len(self.buffer)-1):
            dx = self.buffer[i+1][0] - self.buffer[i][0]
            dy = self.buffer[i+1][1] - self.buffer[i][1]
            distances.append(np.sqrt(dx*dx + dy*dy))
        
        # Check for unreasonable movements
        if len(distances) > 2:
            avg_distance = np.mean(distances[:-1])
            current_distance = distances[-1]
            
            # Only reject truly extreme movements
            if current_distance > 3.0 * avg_distance and avg_distance > 0.01:
                return False
        
        return True

    def update_trajectory(self, x_pixel, y_pixel):
        # Store raw coordinates for perfect alignment
        self.trajectory_path.append((x_pixel, y_pixel))
        if len(self.trajectory_path) > self.max_trajectory_points:
            self.trajectory_path.pop(0)
            
    def get_smoothed_trajectory(self):
        """Minimal smoothing to maintain responsiveness"""
        if len(self.trajectory_path) < 3:
            return self.trajectory_path
            
        # Use very minimal smoothing
        smoothed_path = []
        for i in range(len(self.trajectory_path)):
            if i == 0 or i == len(self.trajectory_path) - 1:
                # Keep first and last points exactly as they are
                smoothed_path.append(self.trajectory_path[i])
            else:
                # Simple 3-point average for middle points
                prev = self.trajectory_path[i-1]
                curr = self.trajectory_path[i]
                next_pt = self.trajectory_path[i+1]
                
                avg_x = (prev[0] + curr[0] + next_pt[0]) / 3
                avg_y = (prev[1] + curr[1] + next_pt[1]) / 3
                
                smoothed_path.append((int(avg_x), int(avg_y)))
            
        return smoothed_path

def re_encode_video_to_h264(input_path, output_path):
    """
    Re-encodes a video to H.264 format using ffmpeg for better browser compatibility.
    """
    command = ['ffmpeg', '-y', '-i', input_path, '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-c:a', 'aac', '-strict', '-2', output_path]
    
    print(f"Attempting to re-encode video: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully re-encoded {input_path} to {output_path}")
        print(f"FFmpeg stdout:\n{result.stdout}")
        print(f"FFmpeg stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg re-encoding (Return Code {e.returncode}): {e.cmd}")
        print(f"FFmpeg stdout:\n{e.stdout}")
        print(f"FFmpeg stderr:\n{e.stderr}")
        print("Re-encoding failed. The output video might not be playable in the browser.")
    except FileNotFoundError:
        print("Error: ffmpeg command not found.")
        print("Please ensure ffmpeg is installed and in your system's PATH.")
        print("Re-encoding skipped.")

def apply_cubic_spline_smoothing(df, lambda_value=0.0005):
    """
    Apply cubic spline smoothing to x and y coordinates
    
    Args:
        df: DataFrame with frame, x, y columns
        lambda_value: Smoothing parameter (higher = more smoothing)
        
    Returns:
        DataFrame with smoothed x and y coordinates
    """
    n = len(df)
    if n < 3:
        return df  # Not enough points for smoothing
    
    # Set up the system for cubic spline with regularization
    t = np.arange(n)
    
    # Identity matrix
    I = np.eye(n)
    
    # Second derivative matrix (finite difference approximation)
    D = np.zeros((n-2, n))
    for i in range(n-2):
        D[i, i] = 1
        D[i, i+1] = -2
        D[i, i+2] = 1
    
    # Smoothing equation: (I + λ*D'D)x = y
    # Where λ is the smoothing parameter
    A = I + lambda_value * D.T @ D
    
    # Solve for smoothed x and y
    x_smoothed = np.linalg.solve(A, df['x'].values)
    y_smoothed = np.linalg.solve(A, df['y'].values)
    
    # Create new dataframe with smoothed values
    smoothed_df = pd.DataFrame({
        'frame': df['frame'].values,
        'x': x_smoothed,
        'y': y_smoothed
    })
    
    return smoothed_df

def process_video(video_path, model_path, output_dir, is_image_sequence=False):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    is_yolov5 = 'v5' in os.path.basename(model_path).lower()  # Detect YOLOv5 by filename
    
    if is_image_sequence:
        # Get all image files in the directory
        image_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        if not image_files:
            raise ValueError("No image files found in the sequence directory")
        
        # Read first image to get dimensions
        first_frame = cv2.imread(os.path.join(video_path, image_files[0]))
        frame_width = first_frame.shape[1]
        frame_height = first_frame.shape[0]
        fps = 30  # Default fps for image sequences
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        detected_video_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_detected.mp4")
        out = cv2.VideoWriter(detected_video_path, fourcc, fps, (frame_width, frame_height))
        
        # Store all frames and detections
        all_frames = []
        all_detections = []
        
        print("Processing image sequence...")
        for img_file in image_files:
            frame = cv2.imread(os.path.join(video_path, img_file))
            if frame is None:
                continue
                
            all_frames.append(frame.copy())
            
            # Run YOLO detection
            results = model.predict(source=frame, imgsz=(1920,1088), conf=0.2)
            
            # Extract all detections with confidence scores
            current_detections = []
            for result in results:
                if hasattr(result, 'boxes') and hasattr(result.boxes, 'xywh'):
                    boxes = result.boxes
                    if len(boxes) > 0:
                        if is_yolov5 and hasattr(boxes, 'cls'):
                            # Only keep detections for class 0 (hammer ball)
                            hammer_ball_indices = (boxes.cls == 0).nonzero(as_tuple=True)[0]
                            if len(hammer_ball_indices) > 0:
                                best_idx = boxes.conf[hammer_ball_indices].argmax().item()
                                idx = hammer_ball_indices[best_idx].item()
                                conf = float(boxes.conf[idx].item())
                                box = boxes.xywh[idx]
                                x_center = float(box[0].item()) / frame_width
                                y_center = float(box[1].item()) / frame_height
                                current_detections = [(x_center, y_center, conf)]
                        else:
                            # Default: take highest confidence detection (for YOLOv8/YOLOv11)
                            best_idx = boxes.conf.argmax().item()
                            conf = float(boxes.conf[best_idx].item())
                            box = boxes.xywh[best_idx]
                            x_center = float(box[0].item()) / frame_width
                            y_center = float(box[1].item()) / frame_height
                            current_detections = [(x_center, y_center, conf)]
            
            all_detections.append(current_detections)
    else:
        # Original video processing code
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        detected_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_detected.mp4")
        out = cv2.VideoWriter(detected_video_path, fourcc, fps, (frame_width, frame_height))
        
        # Store all frames and detections
        all_frames = []
        all_detections = []
        
        print("Processing video frames...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            all_frames.append(frame.copy())
            
            # Run YOLO detection
            results = model.predict(source=frame, imgsz=(1920,1088), conf=0.2)
            
            # Extract all detections with confidence scores
            current_detections = []
            for result in results:
                if hasattr(result, 'boxes') and hasattr(result.boxes, 'xywh'):
                    boxes = result.boxes
                    if len(boxes) > 0:
                        if is_yolov5 and hasattr(boxes, 'cls'):
                            hammer_ball_indices = (boxes.cls == 0).nonzero(as_tuple=True)[0]
                            if len(hammer_ball_indices) > 0:
                                best_idx = boxes.conf[hammer_ball_indices].argmax().item()
                                idx = hammer_ball_indices[best_idx].item()
                                conf = float(boxes.conf[idx].item())
                                box = boxes.xywh[idx]
                                x_center = float(box[0].item()) / frame_width
                                y_center = float(box[1].item()) / frame_height
                                current_detections = [(x_center, y_center, conf)]
                        else:
                            best_idx = boxes.conf.argmax().item()
                            conf = float(boxes.conf[best_idx].item())
                            box = boxes.xywh[best_idx]
                            x_center = float(box[0].item()) / frame_width
                            y_center = float(box[1].item()) / frame_height
                            current_detections = [(x_center, y_center, conf)]
            
            all_detections.append(current_detections)
        
        cap.release()
    
    # Fixed bounding box dimensions
    fixed_width_ratio = 0.016500
    fixed_height_ratio = 0.026800
    fixed_width_pixels = int(fixed_width_ratio * frame_width)
    fixed_height_pixels = int(fixed_height_ratio * frame_height)
    
    # Process frames and draw detections
    tracker = BallTracker()
    results_list = []
    
    print(f"Processing {len(all_frames)} frames with detections...")
    for frame_idx, frame in enumerate(all_frames):
        current_detections = all_detections[frame_idx]
        valid_detection = False
        
        # Initialize default positions
        if tracker.buffer:
            default_x, default_y = tracker.buffer[-1]
        else:
            default_x, default_y = 0.5, 0.5
        
        if current_detections:
            # We already have only the highest confidence detection
            best_detection = current_detections[0]
            x_center, y_center, conf = best_detection
            
            if tracker.validate_detection((x_center, y_center), frame_width, frame_height):
                if not tracker.kf_initialized:
                    tracker.kf.x = np.array([[x_center], [y_center], [0], [0]])
                    tracker.kf_initialized = True
                else:
                    tracker.kf.predict()
                    measurement = np.array([[x_center], [y_center]])
                    tracker.kf.update(measurement)
                
                raw_x = x_center
                raw_y = y_center
                kf_x = float(tracker.kf.x[0, 0])
                kf_y = float(tracker.kf.x[1, 0])
                
                weight_raw = min(1.0, conf * 1.5)
                weight_kf = 1.0 - weight_raw
                
                x_coords = (raw_x * weight_raw) + (kf_x * weight_kf)
                y_coords = (raw_y * weight_raw) + (kf_y * weight_kf)
                
                x_coords = max(0, min(1, x_coords))
                y_coords = max(0, min(1, y_coords))
                
                results_list.append({
                    "frame": frame_idx,
                    "x": x_coords,
                    "y": y_coords,
                    "conf": conf,
                    "raw_x": raw_x,
                    "raw_y": raw_y,
                    "kf_x": kf_x,
                    "kf_y": kf_y
                })
                
                valid_detection = True
                x_pixel = int(x_coords * frame_width)
                y_pixel = int(y_coords * frame_height)
                
                tracker.update_trajectory(x_pixel, y_pixel)
                
                cv2.circle(frame, (x_pixel, y_pixel), radius=5, color=(0, 0, 255), thickness=-1)
                
                top_left = (int(x_pixel - fixed_width_pixels/2), int(y_pixel - fixed_height_pixels/2))
                bottom_right = (int(x_pixel + fixed_width_pixels/2), int(y_pixel + fixed_height_pixels/2))
                cv2.rectangle(frame, top_left, bottom_right, color=(255, 0, 0), thickness=2)
                
                cv2.putText(frame, f'Conf: {conf:.2f}', (x_pixel - 40, y_pixel - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if not valid_detection:
            if tracker.kf_initialized:
                tracker.kf.predict()
                kf_x = float(tracker.kf.x[0, 0])
                kf_y = float(tracker.kf.x[1, 0])
                
                x_coords = max(0, min(1, kf_x))
                y_coords = max(0, min(1, kf_y))
                
                results_list.append({
                    "frame": frame_idx,
                    "x": x_coords,
                    "y": y_coords,
                    "conf": 0.0,
                    "raw_x": default_x,
                    "raw_y": default_y,
                    "kf_x": kf_x,
                    "kf_y": kf_y
                })
            elif tracker.buffer:
                x_coords, y_coords = tracker.buffer[-1]
                results_list.append({
                    "frame": frame_idx,
                    "x": x_coords,
                    "y": y_coords,
                    "conf": 0.0,
                    "raw_x": x_coords,
                    "raw_y": y_coords,
                    "kf_x": x_coords,
                    "kf_y": y_coords
                })
            else:
                results_list.append({
                    "frame": frame_idx,
                    "x": default_x,
                    "y": default_y,
                    "conf": 0.0,
                    "raw_x": default_x,
                    "raw_y": default_y,
                    "kf_x": default_x,
                    "kf_y": default_y
                })
        
        cv2.putText(frame, f'Frame: {frame_idx}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frame_with_trajectory = frame.copy()
        smoothed_path = tracker.get_smoothed_trajectory()
        
        if len(smoothed_path) > 1:
            for i in range(1, len(smoothed_path)):
                cv2.line(frame_with_trajectory, smoothed_path[i-1], smoothed_path[i], 
                        color=(255, 0, 0), thickness=2)
        
        out.write(frame)
    
    out.release()
    
    # Define the path for the re-encoded video in the static directory
    static_detected_path = os.path.join('static', 'detections', os.path.basename(detected_video_path))

    # Re-encode the generated video to H.264
    re_encode_video_to_h264(detected_video_path, static_detected_path)

    # Clean up the original video file
    try:
        os.remove(detected_video_path)
        print(f"Cleaned up original video file: {detected_video_path}")
    except OSError as e:
        print(f"Error removing original video file {detected_video_path}: {e}")

    # Process results into DataFrame
    df = pd.DataFrame(results_list)
    df = df.drop_duplicates(subset='frame').reset_index(drop=True)
    
    # Create a copy with essential columns
    final_df = df[['frame', 'x', 'y', 'conf']].copy()
    
    # Ensure all frames are accounted for
    max_frame = len(all_frames) - 1
    all_frames_df = pd.DataFrame({'frame': range(max_frame + 1)})
    final_df = pd.merge(all_frames_df, final_df, on='frame', how='left')
    
    # Interpolate missing values
    final_df[['x', 'y']] = final_df[['x', 'y']].interpolate(method='linear', limit_direction='both')
    
    # Save results
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = os.path.join(output_dir, f"{base_filename}_coordinates.csv")
    final_df[['frame', 'x', 'y']].to_csv(output_csv, index=False)
    
    # Apply cubic spline smoothing
    smoothed_df = apply_cubic_spline_smoothing(final_df[['frame', 'x', 'y']], lambda_value=0.0005)
    output_smoothed_txt_path = os.path.join(output_dir, f"{base_filename}_smoothed_spline.txt")
    with open(output_smoothed_txt_path, 'w') as f:
        for _, row in smoothed_df.iterrows():
            f.write(f"{row['x']} {row['y']}\n")
    
    return {
        'coordinates': final_df[['frame', 'x', 'y', 'conf']].to_dict(orient='records'),
        'csv_path': output_csv,
        'frame_count': len(all_frames),
        'frame_width': frame_width,
        'frame_height': frame_height,
        'detected_video_path': static_detected_path,
        'smoothed_txt_path': output_smoothed_txt_path,
        'frame_rate': fps
    } 