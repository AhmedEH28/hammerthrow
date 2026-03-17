import os
from .ball_detector import process_video
from .calibration import calibrate_cameras
from .reconstruction import reconstruct_3d_trajectory, analyze_trajectory

def process_throw(side_video_path, back_video_path, model_path, calibration_data, output_dir):
    """
    Process a hammer throw from side and back videos
    
    Args:
        side_video_path: Path to side camera video
        back_video_path: Path to back camera video
        model_path: Path to YOLO model weights
        calibration_data: Dictionary containing calibration data (can be None)
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary containing all analysis results
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    side_output_dir = os.path.join(output_dir, 'side')
    back_output_dir = os.path.join(output_dir, 'back')
    os.makedirs(side_output_dir, exist_ok=True)
    os.makedirs(back_output_dir, exist_ok=True)
    
    # Step 1: Process videos to get 2D coordinates
    print("Processing side video...")
    side_results = process_video(side_video_path, model_path, side_output_dir)
    
    print("Processing back video...")
    back_results = process_video(back_video_path, model_path, back_output_dir)
    
    # If no calibration data, return only 2D detection results
    if calibration_data is None:
        return {
            'side_detection': side_results,
            'back_detection': back_results,
            'message': 'No calibration data provided. Only 2D detection results are available.'
        }
    
    # Step 2: Calibrate cameras
    print("Calibrating cameras...")
    dlt_params = calibrate_cameras(calibration_data)
    
    # Step 3: Reconstruct 3D trajectory
    print("Reconstructing 3D trajectory...")
    frame_analysis = reconstruct_3d_trajectory(
        side_results['coordinates'],
        back_results['coordinates'],
        dlt_params
    )
    
    # Step 4: Analyze trajectory
    print("Analyzing trajectory...")
    analysis_results = analyze_trajectory(frame_analysis)
    
    # Save results
    import json
    results = {
        'side_detection': side_results,
        'back_detection': back_results,
        'dlt_params': {
            'side': dlt_params['side'].tolist(),
            'back': dlt_params['back'].tolist()
        },
        'frame_analysis': frame_analysis,
        'trajectory_analysis': analysis_results
    }
    
    results_path = os.path.join(output_dir, 'analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    # Example usage
    side_video_path = '/path/to/side/video.mp4'
    back_video_path = '/path/to/back/video.mp4'
    model_path = '/path/to/model/weights.pt'
    output_dir = '/path/to/output'
    
    # Example calibration data
    calibration_data = {
        'calibration_3d': [
            [-30, 350, 12510],
            [-30, 350, -12550],
            [12530, 350, 40],
            [-12680, 380, 30],
            [-20180, 190, 22000],
            [-20590, 22410, 21580],
            [-18920, 260, -23360],
            [-19250, 22830, -23380],
            [26540, 470, 20650],
            [26590, 31450, 20950],
            [27600, 480, -20140],
            [27230, 30910, -20680],
            [0, 150, 0]
        ],
        'side_2d': [
            [920.25, 921.00],
            [947.75, 1013.50],
            [610.89, 964.69],
            [1252.00, 957.50],
            [1326.25, 897.25],
            [1326.50, 443.50],
            [1607.25, 1071.75],
            [1610.00, 309.25],
            [361.25, 903.25],
            [365.50, 255.50],
            [47.25, 1062.00],
            [67.50, 53.00],
            [930.75, 965.75]
        ],
        'back_2d': [
            [1233.00, 843.50],
            [585.50, 864.75],
            [885.75, 795.25],
            [978.56, 944.19],
            [1792.00, 987.00],
            [1815.00, 196.00],
            [73.50, 1056.00],
            [57.00, 118.50],
            [1219.00, 741.50],
            [1227.50, 194.50],
            [481.50, 752.50],
            [471.50, 182.50],
            [922.00, 858.00]
        ],
        'image_width': 1920,
        'image_height': 1080
    }
    
    results = process_throw(side_video_path, back_video_path, model_path, 
                          calibration_data, output_dir) 