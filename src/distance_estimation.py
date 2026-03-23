import numpy as np
import warnings

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def apply_height_range_filter(points_3d, min_height=1.7, max_height=2.5):
    """
    Filter points with Y between min_height and max_height
    
    Parameters:
    - points_3d: numpy array of 3D points
    - min_height: minimum height to filter (default 1.7)
    - max_height: maximum height to filter (default 2.5)
    
    Returns:
    - Filtered numpy array of 3D points with original indices
    """
    mask = (points_3d[:, 1] >= min_height) & (points_3d[:, 1] <= max_height)
    # Return filtered points and their original indices to preserve frame numbers
    original_indices = np.arange(len(points_3d))
    return points_3d[mask], original_indices[mask]

def calculate_projectile_motion(points_3d, frame_rate=240, filter_method=None):
    """
    Calculate projectile motion parameters
    
    Parameters:
    - points_3d: numpy array of 3D points
    - frame_rate: frames per second (default 240)
    - filter_method: method of filtering points
      None: no filtering
      'velocity_outliers': remove velocity outliers (applied before calculation)
    
    Returns:
    - List of dictionaries with motion parameters including 3D points and original frame numbers
    """
    # Ensure frame_rate is a float before division
    frame_rate = float(frame_rate)
    delta_t = 1 / frame_rate  # Time interval between frames
    g = 9.81  # Gravitational acceleration (m/s^2)
    results = []
    
    processed_points = points_3d # Start with all points if no method specified initially

    if filter_method == 'velocity_outliers':
        # Filter out velocity outliers BEFORE calculating parameters for the output list
        temp_points_for_velocity_check = []
        original_indices_for_velocity_check = []
        for i in range(len(points_3d) - 1):
             velocity = (points_3d[i + 1] - points_3d[i]) / delta_t
             v_magnitude = np.linalg.norm(velocity)
             
             # Keep points with velocity between 15-35 m/s
             if 15 <= v_magnitude <= 35:
                 temp_points_for_velocity_check.append(points_3d[i]) # Keep the starting point of the segment
                 original_indices_for_velocity_check.append(i) # Keep the index of the starting point

        if temp_points_for_velocity_check:
            processed_points = np.array(temp_points_for_velocity_check)

            # Recalculate parameters for all original points
            all_params = calculate_projectile_motion(points_3d, frame_rate, None) # Calculate without velocity filter first
            
            # Identify velocity outliers from the calculated parameters
            velocities = [p['velocity'] for p in all_params]
            if len(velocities) > 1:
                avg_velocity = np.mean(velocities)
                std_velocity = np.std(velocities)
                # Determine which original frames correspond to velocity outliers
                outlier_frames = {all_params[i]['frame'] for i in range(len(all_params)) if abs(velocities[i] - avg_velocity) > 2 * std_velocity}
                # Filter the original points based on these outlier frames - keep points *not* in outlier frames
                processed_points = np.array([points_3d[i] for i in range(len(points_3d)) if (i + 1) not in outlier_frames])
            else:
                 # If only one or no velocity data points, no outliers can be determined
                 processed_points = points_3d # Keep all original points


def add_average_row(results, skip_first_n_frames=0):
    """
    Add an average row to the results, optionally skipping first N frames.
    
    Parameters:
    - results: List of result dictionaries
    - skip_first_n_frames: Number of initial frames to skip when computing average (default 0)
    
    Returns:
    - List of results with average row appended
    """
    if not results:
        return results
    
    # Filter out frames to skip
    if skip_first_n_frames > 0:
        # Get all unique frame numbers and sort them
        all_frames = sorted(set(r['frame'] for r in results if isinstance(r['frame'], int)))
        if len(all_frames) > skip_first_n_frames:
            frames_to_skip = set(all_frames[:skip_first_n_frames])
            filtered_results = [r for r in results if r['frame'] not in frames_to_skip]
        else:
            filtered_results = results  # Not enough frames to skip
    else:
        filtered_results = results
    
    if not filtered_results:
        return results
    
    # Calculate averages from filtered results
    valid_velocities = [p['velocity'] for p in filtered_results if not np.isnan(p['velocity'])]
    valid_heights = [p['height'] for p in filtered_results if not np.isnan(p['height'])]
    valid_angles = [p['angle'] for p in filtered_results if not np.isnan(p['angle'])]
    valid_distances = [p['distance'] for p in filtered_results if not np.isnan(p['distance'])]
    
    avg_velocity = float(np.mean(valid_velocities)) if valid_velocities else np.nan
    avg_height = float(np.mean(valid_heights)) if valid_heights else np.nan
    avg_angle = float(np.mean(valid_angles)) if valid_angles else np.nan
    avg_distance = float(np.mean(valid_distances)) if valid_distances else np.nan
    
    # Compute distance from averaged parameters (not average of distances)
    g = 9.81
    if not np.isnan(avg_velocity) and not np.isnan(avg_angle) and not np.isnan(avg_height):
        theta = np.radians(avg_angle)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        if avg_velocity > 0:
            sqrt_arg = sin_theta**2 + (2 * g * avg_height) / (avg_velocity**2)
            if sqrt_arg >= 0:
                computed_distance = (avg_velocity**2 * cos_theta / g) * (sin_theta + np.sqrt(sqrt_arg))
            else:
                computed_distance = np.nan
        else:
            computed_distance = np.nan
    else:
        computed_distance = np.nan
    
    # Create average row
    avg_row = {
        "frame": "Average",
        "velocity": avg_velocity,
        "height": avg_height,
        "angle": avg_angle,
        "distance": computed_distance,  # Use computed distance from averaged parameters
        "x": float(np.mean([p['x'] for p in filtered_results])) if filtered_results else np.nan,
        "y": float(np.mean([p['y'] for p in filtered_results])) if filtered_results else np.nan,
        "z": float(np.mean([p['z'] for p in filtered_results])) if filtered_results else np.nan
    }
    
    # Return original results with average row appended
    return results + [avg_row]

# Simplified function to calculate parameters for a given set of 3D points and their frames
def calculate_step_parameters(points_3d_with_frames, frame_rate):
    """
    Calculate projectile motion parameters for a list of 3D points with associated frame numbers.
    
    Parameters:
    - points_3d_with_frames: List of dictionaries [{'frame': int, 'x': float, 'y': float, 'z': float}, ...]
    - frame_rate: frames per second
    
    Returns:
    - List of dictionaries with motion parameters including original frame numbers and 3D points.
    """
    frame_rate = float(frame_rate)
    delta_t = 1 / frame_rate
    g = 9.81
    results = []

    # Sort points by frame number to ensure correct velocity calculation
    sorted_points = sorted(points_3d_with_frames, key=lambda p: p['frame'])

    for i in range(len(sorted_points) - 1):
        p1 = sorted_points[i]
        p2 = sorted_points[i + 1]

        # Calculate velocity between consecutive points
        v_vector = np.array([p2['x'] - p1['x'], p2['y'] - p1['y'], p2['z'] - p1['z']]) / delta_t
        v_magnitude = np.linalg.norm(v_vector)

        # Compute height and release angle (using the second point of the segment as the 'release' point for calculation)
        h_0 = p2['y']

        # Avoid division by zero or issues with zero velocity magnitude
        if v_magnitude**2 == 0:
            theta = np.nan # Angle is undefined if velocity is zero
            throw_distance = np.nan # Distance is undefined if velocity is zero
        else:
            # Velocity vector components
            vx = v_vector[0]
            vy = v_vector[1]
            vz = v_vector[2]

            # Horizontal velocity magnitude
            v_horizontal = np.sqrt(vx**2 + vz**2)

            # Release angle relative to the horizontal plane (XZ plane) using arctan2
            # Handle case where v_horizontal is close to zero
            if v_horizontal == 0:
                 theta = np.pi / 2 if vy > 0 else (-np.pi / 2 if vy < 0 else 0)
            else:
                 theta = np.arctan2(vy, v_horizontal)

            # Calculate sin(theta) and cos(theta)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            # Throw distance calculation using the formula:
            # R = (v^2 * cos(theta) / g) * (sin(theta) + sqrt(sin(theta)^2 + 2*g*h0/v^2))

            # Ensure the argument to np.sqrt is non-negative
            sqrt_arg_dist = sin_theta**2 + (2 * g * h_0) / (v_magnitude**2)

            if sqrt_arg_dist < 0:
                 # This case indicates an issue with the data or formula for this point
                 throw_distance = np.nan
            else:
                 throw_distance = (v_magnitude**2 * cos_theta / g) * (
                     sin_theta + np.sqrt(sqrt_arg_dist)
                 )

        results.append({
            "frame": p2['frame'], # Associate results with the second frame of the segment
            "velocity": v_magnitude,
            "height": h_0,
            "angle": np.degrees(theta) if not np.isnan(theta) else np.nan, # Convert to degrees, handle NaN
            "distance": throw_distance,
            "x": p2['x'],
            "y": p2['y'],
            "z": p2['z']
        })

    return results

# New orchestrator function for the 5-step analysis
def perform_5_step_analysis(points_3d_np, frame_rate):
    """
    Perform the 5-step trajectory analysis on 3D points.
    
    Parameters:
    - points_3d_np: numpy array of 3D points [X, Y, Z]
    - frame_rate: frames per second
    
    Returns:
    - Dictionary containing results for each of the 5 steps.
    """
    # Handle None or invalid frame_rate
    if frame_rate is None:
        print("Warning: frame_rate is None, using default 240 FPS")
        frame_rate = 240.0
    
    try:
        frame_rate = float(frame_rate)
        if frame_rate <= 0:
            print(f"Warning: Invalid frame_rate {frame_rate}, using default 240 FPS")
            frame_rate = 240.0
    except (ValueError, TypeError):
        print(f"Warning: Cannot convert frame_rate {frame_rate} to float, using default 240 FPS")
        frame_rate = 240.0
    
    delta_t = 1 / frame_rate

    # Convert numpy array to list of dicts with frame numbers for easier handling
    points_3d_with_frames = [{'frame': i + 1, 'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])} for i, p in enumerate(points_3d_np)]

    # Step 1: No filtering (All points)
    step1_results = calculate_step_parameters(points_3d_with_frames, frame_rate)
    # Add average row (skip first 3 frames)
    step1_results = add_average_row(step1_results, skip_first_n_frames=3)

    # Step 2: Only height filter (0.5 to 2.50)
    height_filtered_points_2_np, original_indices_2 = apply_height_range_filter(points_3d_np, 1.20, 2.50)
    height_filtered_points_2_with_frames = [{'frame': original_indices_2[i] + 1, 'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])} for i, p in enumerate(height_filtered_points_2_np)]
    step2_results = calculate_step_parameters(height_filtered_points_2_with_frames, frame_rate)
    # Add average row (skip first 3 frames)
    step2_results = add_average_row(step2_results, skip_first_n_frames=1)

    # Step 3: From Step 2's results, filter by velocity (22 to 32 m/s)
    step3_results = []
    for result in step2_results:
        # Skip the average row when filtering
        if result['frame'] == "Average":
            continue
        if result['velocity'] is not None and 20 <= result['velocity'] <= 30:
            step3_results.append(result)
    # Add average row (no skipping)
    step3_results = add_average_row(step3_results, skip_first_n_frames=0)

    step4_results = []

    # Height filter (same bounds as used in Step 2)
    height_filtered_points_4_np, original_indices_4 = apply_height_range_filter(points_3d_np, 1.20, 2.50)
    if height_filtered_points_4_np.size > 0:
        # Prepare points with frame indices
        height_filtered_with_frames = [
            {'frame': int(original_indices_4[i]) + 1, 'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])}
            for i, p in enumerate(height_filtered_points_4_np)
        ]

        # Calculate per-segment parameters for the height-filtered points
        step4_candidate_params = calculate_step_parameters(height_filtered_with_frames, frame_rate)

        # Velocity filter: keep segments with velocity in the same range used in Step 3
        velocity_filtered = [r for r in step4_candidate_params if r['velocity'] is not None and 20 <= r['velocity'] <= 30]

        # Angle filter: now filter by angle range (30 to 50 degrees)
        angle_filtered_results = [r for r in velocity_filtered if r['angle'] is not None and 30 <= r['angle'] <= 50]

        # Use the final filtered results as Step 4 data rows
        if angle_filtered_results:
            step4_results = angle_filtered_results
            # Append an average row for Step 4 (no skipping)
            step4_results = add_average_row(step4_results, skip_first_n_frames=0)

    # Step 5: Compute averages (velocity, height, angle) from Step 4 and compute final distance
    step5_results = []
    if step4_results and len(step4_results) > 0:
        # Filter out the average row from step4 to get only data rows
        step4_data_rows = [r for r in step4_results if r['frame'] != "Average"]
        
        if step4_data_rows:
            # Calculate average values from Step 4 data rows
            valid_velocities = [p['velocity'] for p in step4_data_rows if not np.isnan(p['velocity'])]
            valid_heights = [p['height'] for p in step4_data_rows if not np.isnan(p['height'])]
            valid_angles = [p['angle'] for p in step4_data_rows if not np.isnan(p['angle'])]

            if not valid_velocities:
                avg_velocity = np.nan
            else:
                avg_velocity = float(np.mean(valid_velocities))

            avg_height = float(np.mean(valid_heights)) if valid_heights else np.nan
            avg_angle = float(np.mean(valid_angles)) if valid_angles else np.nan

            # Compute final throw distance from averaged parameters
            g = 9.81
            if np.isnan(avg_velocity) or np.isnan(avg_angle):
                final_distance = np.nan
            else:
                theta = np.radians(avg_angle)
                sin_theta = np.sin(theta)
                cos_theta = np.cos(theta)
                # Protect against division by zero
                if avg_velocity == 0:
                    final_distance = np.nan
                else:
                    sqrt_arg = sin_theta**2 + (2 * g * avg_height) / (avg_velocity**2) if not np.isnan(avg_height) else sin_theta**2
                    if sqrt_arg < 0:
                        final_distance = np.nan
                    else:
                        final_distance = (avg_velocity**2 * cos_theta / g) * (sin_theta + np.sqrt(sqrt_arg))

            # Create a single result with average values and computed final distance
            step5_results = [{
                "frame": "Average",
                "velocity": avg_velocity,
                "height": avg_height,
                "angle": avg_angle,
                "distance": final_distance,
                "x": float(np.mean([p['x'] for p in step4_data_rows])) if step4_data_rows else np.nan,
                "y": float(np.mean([p['y'] for p in step4_data_rows])) if step4_data_rows else np.nan,
                "z": float(np.mean([p['z'] for p in step4_data_rows])) if step4_data_rows else np.nan
            }]


    return {
        'results_analysis_after_outlier_filtered': step4_results,
        'step4_velocity_outliers': step4_results,
        'step5_recalculated_avg_vel': step5_results
    }


def analyze_trajectory(frame_analysis, frame_rate=240):
    """
    Analyze trajectory and return summary statistics
    
    Parameters:
    - frame_analysis: Dictionary containing analysis results
    - frame_rate: frames per second (default 240)
    
    Returns:
    - Dictionary containing summary statistics
    """
    return {'message': 'Detailed analysis results available in 5 steps.'}

def estimate_throw_distance(points_3d, frame_rate=240):
    """
    Estimate throw distance using multiple filtering methods
    
    Parameters:
    - points_3d: numpy array of 3D points
    - frame_rate: frames per second (default 240)
    
    Returns:
    - Dictionary containing results from different filtering methods
    """
    # This function will be replaced by the call to perform_5_step_analysis
    return {} 