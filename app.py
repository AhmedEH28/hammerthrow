from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify, session
import os
import uuid
from werkzeug.utils import secure_filename
import sys
import json
import math
import numpy as np
import shutil

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.main import process_video
from src.calibration import calibrate_cameras
from src.reconstruction import reconstruct_3d_trajectory
from src.distance_estimation import perform_5_step_analysis
from src.test_throw_manager import TestThrowManager

app = Flask(__name__, template_folder='hammerindex')
app.secret_key = os.urandom(24)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DETECTION_FOLDER'] = 'static/detections'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MODEL_FOLDER'] = 'models'
app.config['TEST_THROWS_FOLDER'] = 'test_throws'  # New: folder containing test throw data
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 megabytes
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
ALLOWED_CALIBRATION_EXTENSIONS = {'csv', 'txt'}

# Initialize test throw manager
test_throw_manager = TestThrowManager(app.config['TEST_THROWS_FOLDER'])

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['DETECTION_FOLDER'], 
               app.config['RESULTS_FOLDER'], app.config['MODEL_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

def replace_nan_with_none(data):
    """Recursively replace NaN float values with None in dictionaries and lists."""
    if isinstance(data, dict):
        return {k: replace_nan_with_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_nan_with_none(item) for item in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    elif isinstance(data, np.integer):
        return int(data)
    else:
        return data

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def is_image_sequence(files):
    """Check if the uploaded files form an image sequence"""
    if not files:
        return False
    # Check if all files are images
    return all(allowed_file(f.filename, ALLOWED_IMAGE_EXTENSIONS) for f in files)

def parse_calibration_file(file_path):
    """Parse calibration file to extract 3D and 2D points"""
    import pandas as pd
    
    # Read calibration data
    df = pd.read_csv(file_path)
    
    # Extract columns
    calibration_3d = df[['X', 'Y', 'Z']].values
    side_2d = df[['x1', 'y1']].values
    back_2d = df[['x2', 'y2']].values
    
    return {
        'calibration_3d': calibration_3d.tolist(),
        'side_2d': side_2d.tolist(),
        'back_2d': back_2d.tolist(),
        'image_width': 1920,
        'image_height': 1080
    }

@app.route('/', methods=['GET'])
def index():
    """Serve the advanced analysis interface at the root URL by default"""
    # Make the advanced interface the default served at '/'
    return render_template('index.html', results={})

@app.route('/advanced', methods=['GET'])
def advanced_analysis():
    """Serve the original advanced interface"""
    return render_template('index.html', results={})

@app.route('/simple', methods=['GET'])
def simple_analysis():
    """Serve the simplified analysis interface (same as default)"""
    return render_template('simple_analysis.html')

@app.route('/upload_test_throw', methods=['POST'])
def upload_test_throw():
    """
    Upload a complete test throw folder with all necessary files
    """
    try:
        # Get uploaded files
        side_video = request.files.get('side_video')
        back_video = request.files.get('back_video')
        calibration_file = request.files.get('calibration_file')
        metadata_file = request.files.get('metadata_file')  # Optional
        throw_name = request.form.get('throw_name', 'uploaded_throw')
        
        if not side_video or not back_video or not calibration_file:
            return jsonify({'error': 'Side video, back video, and calibration file are required'}), 400
        
        # Create unique throw folder
        throw_folder = os.path.join(app.config['TEST_THROWS_FOLDER'], f"{throw_name}_{uuid.uuid4().hex[:8]}")
        os.makedirs(throw_folder, exist_ok=True)
        
        # Save files
        side_filename = f"side_view.{side_video.filename.rsplit('.', 1)[1].lower()}"
        back_filename = f"back_view.{back_video.filename.rsplit('.', 1)[1].lower()}"
        calib_filename = f"calibration.{calibration_file.filename.rsplit('.', 1)[1].lower()}"
        
        side_path = os.path.join(throw_folder, side_filename)
        back_path = os.path.join(throw_folder, back_filename)
        calib_path = os.path.join(throw_folder, calib_filename)
        
        side_video.save(side_path)
        back_video.save(back_path)
        calibration_file.save(calib_path)
        
        # Save metadata if provided
        if metadata_file:
            metadata_filename = f"metadata.{metadata_file.filename.rsplit('.', 1)[1].lower()}"
            metadata_path = os.path.join(throw_folder, metadata_filename)
            metadata_file.save(metadata_path)
        else:
            # Create metadata from form data if provided
            release_point = request.form.get('release_point')
            frame_rate = request.form.get('frame_rate')
            
            if release_point or frame_rate:
                metadata_data = {}
                if release_point:
                    try:
                        metadata_data['release_point'] = int(release_point)
                    except ValueError:
                        pass
                if frame_rate:
                    try:
                        metadata_data['frame_rate'] = float(frame_rate)
                    except ValueError:
                        pass
                
                if metadata_data:
                    metadata_path = os.path.join(throw_folder, 'metadata.json')
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata_data, f, indent=2)
        
        # Validate the uploaded throw
        validation = test_throw_manager.validate_test_throw_folder(throw_folder)
        
        return jsonify({
            'success': True,
            'throw_folder': os.path.basename(throw_folder),
            'validation': validation,
            'message': f'Test throw "{throw_name}" uploaded successfully!'
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/get_available_test_throws', methods=['GET'])
def get_available_test_throws():
    """Get list of available test throw folders"""
    try:
        test_throws = test_throw_manager.get_available_test_throws()
        return jsonify({'test_throws': test_throws})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_test_throw', methods=['POST'])
def analyze_test_throw():
    """
    Complete analysis of a test throw from folder selection to final results
    """
    try:
        data = request.json
        test_throw_name = data.get('test_throw_name')
        model_name = data.get('model_name')
        
        if not test_throw_name or not model_name:
            return jsonify({'error': 'Test throw name and model name are required'}), 400
        
        # Get test throw data
        throw_data = test_throw_manager.get_test_throw_data(test_throw_name)
        if not throw_data:
            return jsonify({'error': f'Test throw {test_throw_name} not found'}), 400
        
        # Validate test throw folder
        validation = test_throw_manager.validate_test_throw_folder(throw_data['path'])
        if not validation['valid']:
            return jsonify({'error': f'Invalid test throw folder: {"; ".join(validation["errors"])}'}), 400
        
        # Process videos for 2D detection
        model_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_name}.pt")
        
        print(f"Processing side video: {throw_data['side_video']}")
        side_results = process_video(throw_data['side_video'], model_path, app.config['RESULTS_FOLDER'])
        
        print(f"Processing back video: {throw_data['back_video']}")
        back_results = process_video(throw_data['back_video'], model_path, app.config['RESULTS_FOLDER'])
        
        # Parse calibration file
        calibration_dict = parse_calibration_file(throw_data['calibration_file'])
        
        # Calculate DLT parameters
        dlt_params = calibrate_cameras(calibration_dict)
        
        # Get frame rate and release point
        frame_rate = throw_data.get('frame_rate')
        if not frame_rate:
            # Try to get from video results
            frame_rate = side_results.get('frame_rate')
        if not frame_rate:
            # Use default frame rate
            frame_rate = 240.0
            print(f"Warning: No frame rate found in metadata or video, using default: {frame_rate}")
        
        # Ensure frame_rate is a float
        try:
            frame_rate = float(frame_rate)
        except (ValueError, TypeError):
            frame_rate = 240.0
            print(f"Warning: Invalid frame rate, using default: {frame_rate}")
        
        release_point = throw_data.get('release_point')
        
        # Calculate frame range for analysis
        if release_point:
            start_frame, end_frame = test_throw_manager.calculate_frame_range(release_point)
            print(f"Using frame range: {start_frame} to {end_frame} (based on release point {release_point})")
        else:
            # Use default range: 30 frames before the last 50 frames
            total_frames = len(side_results.get('coordinates', []))
            start_frame, end_frame = test_throw_manager.calculate_default_frame_range(total_frames)
            print(f"Using default frame range: {start_frame} to {end_frame} (30 frames before last 50 frames, total frames: {total_frames})")
        
        # Filter coordinates for the selected frame range
        side_coords_filtered = [coord for coord in side_results.get('coordinates', []) 
                               if start_frame <= coord['frame'] <= end_frame]
        back_coords_filtered = [coord for coord in back_results.get('coordinates', []) 
                               if start_frame <= coord['frame'] <= end_frame]
        
        if not side_coords_filtered or not back_coords_filtered:
            return jsonify({'error': 'No coordinates found in the specified frame range'}), 400
        
        # Align coordinates by frame
        aligned_coords = align_coordinates_by_frame(side_coords_filtered, back_coords_filtered)
        if not aligned_coords:
            return jsonify({'error': 'No matching frames found between side and back views'}), 400
        
        # Prepare coordinates for 3D reconstruction
        side_coords_for_3d = [{'frame': item['frame'], 'x': item['side']['x'], 'y': item['side']['y']} 
                             for item in aligned_coords]
        back_coords_for_3d = [{'frame': item['frame'], 'x': item['back']['x'], 'y': item['back']['y']} 
                             for item in aligned_coords]
        
        # 3D reconstruction
        reconstruction_results = reconstruct_3d_trajectory(
            side_coords_for_3d,
            back_coords_for_3d,
            {'side': dlt_params['side'], 'back': dlt_params['back']},
            frame_rate
        )
        
        points_3d = reconstruction_results.get('points_3d', [])
        if not points_3d:
            return jsonify({'error': 'Failed to reconstruct 3D points'}), 400
        
        # Convert to numpy array for analysis
        if isinstance(points_3d, list):
            points_3d_np = np.array(points_3d, dtype=float)
        else:
            points_3d_np = points_3d
        
        # Debug: Check frame rate before analysis
        print(f"DEBUG: frame_rate before perform_5_step_analysis: {frame_rate} (type: {type(frame_rate)})")
        
        # Perform 5-step analysis with final validation
        if frame_rate is None:
            print("WARNING: frame_rate is None, setting to default 240.0")
            frame_rate = 240.0
        analysis_results = perform_5_step_analysis(points_3d_np, frame_rate)

        # Reduce analysis_results for the simplified UI: keep only the outlier-filtered results and the averaged step.
        # Also expose a clearer key name for the outlier-filtered results for display purposes.
        filtered_analysis = {}
        # The analysis function now returns the reduced keys; handle both old and new key names
        step4 = analysis_results.get('results_analysis_after_outlier_filtered', []) or analysis_results.get('step4_velocity_outliers', [])
        step5 = analysis_results.get('step5_recalculated_avg_vel', [])
        # Ensure step4 is present under the UI-friendly key name
        filtered_analysis['results_analysis_after_outlier_filtered'] = step4
        filtered_analysis['step5_recalculated_avg_vel'] = step5
        
        # Filter out warnings that are no longer relevant since we successfully extracted metadata
        final_warnings = []
        for warning in validation.get('warnings', []):
            # Skip the metadata warning if we successfully extracted metadata
            if "No metadata file found" in warning and (release_point is not None or frame_rate != 240.0):
                print(f"DEBUG: Filtering out warning: {warning}")
                continue
            final_warnings.append(warning)
        
        print(f"DEBUG: Original warnings: {validation.get('warnings', [])}")
        print(f"DEBUG: Final warnings: {final_warnings}")
        print(f"DEBUG: Release point: {release_point}, Frame rate: {frame_rate}")
        
        # Prepare final results
        final_results = {
            'test_throw_name': test_throw_name,
            'model_used': model_name,
            'frame_range': {'start': start_frame, 'end': end_frame},
            'release_point': release_point,
            'frame_rate': frame_rate,
            'points_analyzed': len(points_3d),
            'side_detected_video': side_results.get('detected_video_path', '').replace('results/', '/static/detections/'),
            'back_detected_video': back_results.get('detected_video_path', '').replace('results/', '/static/detections/'),
            'analysis_results': replace_nan_with_none(filtered_analysis),
            'validation_warnings': final_warnings
        }
        
        return jsonify(final_results)
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def align_coordinates_by_frame(side_coords: list, back_coords: list) -> list:
    """
    Align coordinates by frame number
    
    Args:
        side_coords: List of side view coordinates
        back_coords: List of back view coordinates
        
    Returns:
        List of aligned coordinates
    """
    aligned_coordinates = []
    back_frames = {coord['frame']: coord for coord in back_coords}
    
    for side_coord in side_coords:
        frame_num = side_coord['frame']
        if frame_num in back_frames:
            back_coord = back_frames[frame_num]
            aligned_coordinates.append({
                'frame': frame_num,
                'side': {'x': side_coord['x'], 'y': side_coord['y']},
                'back': {'x': back_coord['x'], 'y': back_coord['y']}
            })
    
    return aligned_coordinates

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get uploaded files and model name from the form data
        side_data = request.files.getlist('side_data')
        back_data = request.files.getlist('back_data')
        model_name = request.form['model']

        if not side_data or not back_data:
            return jsonify({'error': 'Side and back view files are required'}), 400

        # Check if we're dealing with image sequences
        is_side_sequence = is_image_sequence(side_data)
        is_back_sequence = is_image_sequence(back_data)

        if is_side_sequence and is_back_sequence:
            # Handle image sequences
            if len(side_data) != len(back_data):
                return jsonify({'error': 'Side and back view must have the same number of frames'}), 400
            
            # Create temporary directories for the sequences
            side_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"side_{uuid.uuid4()}")
            back_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"back_{uuid.uuid4()}")
            os.makedirs(side_dir, exist_ok=True)
            os.makedirs(back_dir, exist_ok=True)

            # Save images in order
            for i, (side_img, back_img) in enumerate(zip(side_data, back_data)):
                side_path = os.path.join(side_dir, f"frame_{i:04d}.jpg")
                back_path = os.path.join(back_dir, f"frame_{i:04d}.jpg")
                side_img.save(side_path)
                back_img.save(back_path)

            # Process the image sequences
            model_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_name}.pt")
            side_results = process_video(side_dir, model_path, app.config['RESULTS_FOLDER'], is_image_sequence=True)
            back_results = process_video(back_dir, model_path, app.config['RESULTS_FOLDER'], is_image_sequence=True)

            # Clean up temporary directories
            shutil.rmtree(side_dir)
            shutil.rmtree(back_dir)

        else:
            # Handle single video files
            if len(side_data) != 1 or len(back_data) != 1:
                return jsonify({'error': 'Please upload either video files or image sequences'}), 400
            
            side_data = side_data[0]
            back_data = back_data[0]
            
            if not allowed_file(side_data.filename, ALLOWED_VIDEO_EXTENSIONS):
                return jsonify({'error': 'Invalid file format for side view'}), 400
            if not allowed_file(back_data.filename, ALLOWED_VIDEO_EXTENSIONS):
                return jsonify({'error': 'Invalid file format for back view'}), 400

            side_filename = f"{uuid.uuid4()}_{secure_filename(side_data.filename)}"
            back_filename = f"{uuid.uuid4()}_{secure_filename(back_data.filename)}"
            side_path = os.path.join(app.config['UPLOAD_FOLDER'], side_filename)
            back_path = os.path.join(app.config['UPLOAD_FOLDER'], back_filename)
            side_data.save(side_path)
            back_data.save(back_path)
            model_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_name}.pt")
            
            # Run detection
            side_results = process_video(side_path, model_path, app.config['RESULTS_FOLDER'])
            back_results = process_video(back_path, model_path, app.config['RESULTS_FOLDER'])
            
            # Clean up uploaded files
            os.remove(side_path)
            os.remove(back_path)

        # Save 2D results to file for later use
        detect_id = str(uuid.uuid4())
        detect_results_path = os.path.join(app.config['RESULTS_FOLDER'], f"detect_{detect_id}.json")
        with open(detect_results_path, 'w') as f:
            json.dump({'side': side_results, 'back': back_results}, f)

        return jsonify({
            'detect_id': detect_id,
            'side_results': side_results,
            'back_results': back_results,
            'frame_rate': side_results['frame_rate'],
            'side_detected_video': side_results['detected_video_path'],
            'back_detected_video': back_results['detected_video_path']
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/calculate_dlt', methods=['POST'])
def calculate_dlt():
    try:
        calibration_data = request.files.get('calibration_data')
        if not calibration_data:
            return jsonify({'error': 'Calibration file is required'}), 400
        if not allowed_file(calibration_data.filename, ALLOWED_CALIBRATION_EXTENSIONS):
            return jsonify({'error': 'Invalid calibration file format'}), 400

        # Save the uploaded calibration file temporarily
        calibration_filename = f"{uuid.uuid4()}_{secure_filename(calibration_data.filename)}"
        calibration_path = os.path.join(app.config['UPLOAD_FOLDER'], calibration_filename)
        calibration_data.save(calibration_path)

        # Parse the calibration file and calculate DLT parameters
        calibration_dict = parse_calibration_file(calibration_path)
        dlt_params = calibrate_cameras(calibration_dict)

        # Clean up the temporary calibration file
        os.remove(calibration_path)

        return jsonify({
            'dlt_params': dlt_params,
            'calibration_data': calibration_dict
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get data from the request body (JSON)
        data = request.json

        # --- Add logging for received data ---
        import json
        print("Received data for /analyze:")
        print(json.dumps(data, indent=2))
        # ---------------------------------------

        action = data.get('action') # Get the action parameter (e.g., 'reconstruct', 'analyze')
        frame_rate = data.get('frame_rate')

        if frame_rate is None:
            return jsonify({'error': 'Frame rate is required'}), 400

        try:
            frame_rate = float(frame_rate)
        except ValueError:
            return jsonify({'error': 'Invalid frame rate format'}), 400

        if action == 'reconstruct':
            # Handle 3D Reconstruction Request
            side_coordinates = data.get('side_coordinates')
            back_coordinates = data.get('back_coordinates')
            dlt_params = {
                'side': data.get('side_dlt_params'),
                'back': data.get('back_dlt_params')
            }

            # Check if required 2D data is present for reconstruction
            if not side_coordinates or not back_coordinates or not dlt_params['side'] or not dlt_params['back']:
                return jsonify({'error': 'Side and back coordinates and DLT parameters are required for 3D reconstruction'}), 400

            # Ensure all numerical data is float
            try:
                # Convert 2D coordinates to floats
                # Assume side_coordinates and back_coordinates are already lists of {'frame': int, 'x': float, 'y': float}
                # from the frontend preprocessing (e.g., alignCoordinatesByFrame result)
                side_coordinates = [{'frame': point['frame'], 'x': float(point['x']), 'y': float(point['y'])} for point in side_coordinates]
                back_coordinates = [{'frame': point['frame'], 'x': float(point['x']), 'y': float(point['y'])} for point in back_coordinates]

                # Convert DLT parameters to floats
                dlt_params['side'] = [float(p) for p in dlt_params['side']]
                dlt_params['back'] = [float(p) for p in dlt_params['back']]
            except (ValueError, KeyError) as e:
                return jsonify({'error': f'Invalid numerical data format in 2D points or DLT parameters: {e}'}), 400

            # Run 3D reconstruction
            print("Calling reconstruct_3d_trajectory...")
            reconstruction_results = reconstruct_3d_trajectory(
                side_coordinates,
                back_coordinates,
                dlt_params,
                frame_rate
            )
            print("reconstruct_3d_trajectory returned.")

            # The reconstruction_results will contain 'points_3d' (numpy array) and potentially 'reprojection_errors'
            # Convert points_3d numpy array to a list of lists for JSON response
            points_3d_data = reconstruction_results.get('points_3d')
            if points_3d_data is not None and isinstance(points_3d_data, np.ndarray):
                points_3d_list = points_3d_data.tolist()
            elif isinstance(points_3d_data, list): # If it's already a list, use it directly
                 points_3d_list = points_3d_data
            else:
                 # If it's None or some other unexpected type
                 points_3d_list = []
                 if points_3d_data is not None:
                     print(f"Warning: reconstruction_results['points_3d'] is not a numpy array or list: {type(points_3d_data)}")

            reprojection_errors = reconstruction_results.get('reprojection_errors', {})

            # Return only the 3D points and reprojection errors for the reconstruction step
            return jsonify({
                'points_3d': points_3d_list,
                'reprojection_errors': reprojection_errors
            })

        elif action == 'analyze':
            # Handle 5-Step Analysis Request
            points_3d = data.get('points_3d') # Expecting 3D points directly

            if not points_3d:
                 return jsonify({'error': '3D points are required for analysis'}), 400

            # Ensure 3D points are in the correct format (list of lists or list of dicts) and convert to numpy array
            points_3d_np = None
            if isinstance(points_3d, list) and len(points_3d) > 0:
                if isinstance(points_3d[0], list) and len(points_3d[0]) == 3:
                     try:
                         points_3d_np = np.array(points_3d, dtype=float)
                     except ValueError:
                         return jsonify({'error': 'Invalid numerical data in 3D points (list format)'}), 400
                elif isinstance(points_3d[0], dict) and 'x' in points_3d[0] and 'y' in points_3d[0] and 'z' in points_3d[0]:
                     try:
                         points_3d_np = np.array([[p['x'], p['y'], p['z']] for p in points_3d], dtype=float)
                     except ValueError:
                          return jsonify({'error': 'Invalid numerical data in 3D points (from dict format)'}), 400
                else:
                     return jsonify({'error': 'Unsupported 3D points data format for analysis'}), 400
            else:
                 return jsonify({'error': 'Invalid 3D points data format for analysis'}), 400

            # Perform 5-step analysis on the provided 3D points
            print("Calling perform_5_step_analysis...")
            analysis_results_5_steps = perform_5_step_analysis(points_3d_np, frame_rate)
            print("perform_5_step_analysis returned.")

            # Replace NaN values with None before sending as JSON
            analysis_results_5_steps = replace_nan_with_none(analysis_results_5_steps)

            # Return the results of the 5-step analysis
            return jsonify(analysis_results_5_steps)

        else:
            # Handle missing or invalid action
            return jsonify({'error': 'Invalid or missing action parameter'}), 400

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/detections/<filename>')
def send_detection(filename):
    return send_from_directory(app.config['DETECTION_FOLDER'], filename)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', error="Page not found"), 404

if __name__ == '__main__':
    app.run(debug=True)