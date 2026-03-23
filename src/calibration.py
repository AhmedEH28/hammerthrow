import numpy as np
import pandas as pd

def compute_dlt_parameters(calibration_3d, points_2d, image_width, image_height):
    """
    Compute DLT parameters for camera calibration
    
    Args:
        calibration_3d: 3D points (scaled by ScaleFactor=10000)
        points_2d: 2D pixel coordinates
        image_width: Image width (1920)
        image_height: Image height (1080)
        
    Returns:
        dlt_params: 11 DLT parameters (L1-L11)
    """
    # Convert 3D points to meters (scale by 1/10000)
    points_3d = calibration_3d / 10000.0
    
    # Normalize 2D points to [0, 1]
    u = points_2d[:, 0] / image_width
    v = points_2d[:, 1] / image_height
    
    n_points = points_3d.shape[0]
    
    # Build design matrix A and vector b
    A = []
    b = []
    for i in range(n_points):
        X, Y, Z = points_3d[i]
        ui = u[i]
        vi = v[i]
        
        # Equation for u: L1*X + L2*Y + L3*Z + L4 - ui*(L9*X + L10*Y + L11*Z) = ui
        row_u = [X, Y, Z, 1, 0, 0, 0, 0, -ui*X, -ui*Y, -ui*Z]
        A.append(row_u)
        b.append(ui)
        
        # Equation for v: L5*X + L6*Y + L7*Z + L8 - vi*(L9*X + L10*Y + L11*Z) = vi
        row_v = [0, 0, 0, 0, X, Y, Z, 1, -vi*X, -vi*Y, -vi*Z]
        A.append(row_v)
        b.append(vi)
    
    A = np.array(A)
    b = np.array(b)
    
    # Solve using least squares
    dlt_params, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    
    return dlt_params

def compute_dlt_parameters_simi(calibration_3d, points_2d, image_width, image_height):
    """
    Compute DLT parameters matching Simi's approach.
    
    Args:
        calibration_3d: 3D points (scaled by ScaleFactor=10000)
        points_2d: 2D pixel coordinates
        image_width: Image width (1920)
        image_height: Image height (1080)
        
    Returns:
        dlt_params: 11 DLT parameters (L1-L11)
    """
    # Convert 3D points to meters (scale by 1/10000)
    points_3d = np.array(calibration_3d) / 10000.0
    
    # Normalize 2D points to [0, 1]
    u = np.array(points_2d)[:, 0] / image_width
    v = np.array(points_2d)[:, 1] / image_height
    
    n_points = points_3d.shape[0]
    
    # Build design matrix A and vector b
    A = []
    b = []
    for i in range(n_points):
        X, Y, Z = points_3d[i]
        ui = u[i]
        vi = v[i]
        
        # Equation for u: L1*X + L2*Y + L3*Z + L4 - ui*(L9*X + L10*Y + L11*Z) = ui
        row_u = [X, Y, Z, 1, 0, 0, 0, 0, -ui*X, -ui*Y, -ui*Z]
        A.append(row_u)
        b.append(ui)
        
        # Equation for v: L5*X + L6*Y + L7*Z + L8 - vi*(L9*X + L10*Y + L11*Z) = vi
        row_v = [0, 0, 0, 0, X, Y, Z, 1, -vi*X, -vi*Y, -vi*Z]
        A.append(row_v)
        b.append(vi)
    
    A = np.array(A)
    b = np.array(b)
    
    # --- Debugging: Log DLT input data ---
    print("DEBUG: Input points_3d for DLT calculation:")
    print(points_3d)
    print("DEBUG: Input u (normalized 2D x) for DLT calculation:")
    print(u)
    print("DEBUG: Input v (normalized 2D y) for DLT calculation:")
    print(v)
    print("-------------------------------------")
    # ---------------------------------------
    
    dlt_params, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    
    return dlt_params

def calibrate_cameras(calibration_data):
    """
    Calibrate both side and back cameras using DLT
    
    Args:
        calibration_data: Dictionary containing:
            - calibration_3d: 3D calibration points
            - side_2d: 2D points for side camera
            - back_2d: 2D points for back camera
            - image_width: Image width (default 1920)
            - image_height: Image height (default 1080)
    
    Returns:
        Dictionary containing DLT parameters for both cameras
    """
    # Extract data
    calibration_3d = calibration_data['calibration_3d']
    side_2d = calibration_data['side_2d']
    back_2d = calibration_data['back_2d']
    image_width = calibration_data.get('image_width', 1920)
    image_height = calibration_data.get('image_height', 1080)
    
    # Compute DLT parameters for both cameras using the new function
    dlt_side = compute_dlt_parameters_simi(calibration_3d, side_2d, image_width, image_height)
    dlt_back = compute_dlt_parameters_simi(calibration_3d, back_2d, image_width, image_height)
    
    return {
        'side': dlt_side.tolist(),
        'back': dlt_back.tolist()
    }

def parse_calibration_file(file_path):
    """
    Parse a CSV file containing calibration data and return a dictionary with the data
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        Dictionary containing:
            - calibration_3d: 3D calibration points
            - side_2d: 2D points for side camera
            - back_2d: 2D points for back camera
            - image_width: Image width (default 1920)
            - image_height: Image height (default 1080)
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Extract columns
    calibration_3d = df[['X', 'Y', 'Z']].values
    side_2d = df[['x1', 'y1']].values
    back_2d = df[['x2', 'y2']].values

    return {
        'calibration_3d': calibration_3d,
        'side_2d': side_2d,
        'back_2d': back_2d,
        'image_width': 1920,
        'image_height': 1080
    } 