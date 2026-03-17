import numpy as np
import warnings
from scipy.interpolate import CubicSpline
from .distance_estimation import estimate_throw_distance, analyze_trajectory
import pandas as pd

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def extract_camera_matrix_from_dlt(dlt_params):
    """
    Extract 3x4 camera matrix from 11 DLT parameters
    """
    L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11 = dlt_params
    
    # Build camera matrix
    P = np.array([
        [L1, L2, L3, L4],
        [L5, L6, L7, L8],
        [L9, L10, L11, 1]
    ])
    
    return P

def triangulate_points(points_2d_back, points_2d_side, dlt_back, dlt_side):
    """
    Reconstruct 3D points from 2D coordinates using DLT parameters
    """
    # Extract camera matrices
    P_back = extract_camera_matrix_from_dlt(dlt_back)
    P_side = extract_camera_matrix_from_dlt(dlt_side)
    
    # Triangulate each point
    points_3d = []
    for p_back, p_side in zip(points_2d_back, points_2d_side):
        # Build system of equations
        A = np.zeros((4, 4))
        A[0] = p_back[0] * P_back[2] - P_back[0]
        A[1] = p_back[1] * P_back[2] - P_back[1]
        A[2] = p_side[0] * P_side[2] - P_side[0]
        A[3] = p_side[1] * P_side[2] - P_side[1]
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        point_3d = Vt[-1, :3] / Vt[-1, 3]
        points_3d.append(point_3d)
    
    return np.array(points_3d)

def normalize_points(points):
    """
    Normalize coordinates to improve numerical stability
    """
    return np.array(points) / 10000.0

def detect_outliers(points, threshold=2.0):
    """
    Detect outliers in 3D points using Mahalanobis distance
    Args:
        points: numpy array of shape (n, 3) containing 3D points
        threshold: number of standard deviations to consider as outlier
    Returns:
        mask: boolean array indicating which points are not outliers
    """
    # Calculate mean and covariance
    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)
    
    # Calculate Mahalanobis distance for each point
    diff = points - mean
    inv_cov = np.linalg.inv(cov)
    mahalanobis = np.sqrt(np.sum(diff.dot(inv_cov) * diff, axis=1))
    
    # Points with distance > threshold * std are outliers
    return mahalanobis < (threshold * np.std(mahalanobis))

def smooth_3d_points(points, lambda_value=0.0005):
    """
    Apply cubic spline smoothing to 3D points
    Args:
        points: numpy array of shape (n, 3) containing 3D points
        lambda_value: smoothing parameter
    Returns:
        smoothed_points: numpy array of smoothed 3D points
    """
    n = len(points)
    if n < 3:
        return points

    # Create time points
    t = np.arange(n)
    
    # Identity matrix
    I = np.eye(n)
    
    # Second derivative matrix
    D = np.zeros((n-2, n))
    for i in range(n-2):
        D[i, i] = 1
        D[i, i+1] = -2
        D[i, i+2] = 1
    
    # Smoothing equation: (I + λ*D'D)x = y
    A = I + lambda_value * D.T @ D
    
    # Apply smoothing to each dimension
    smoothed_points = np.zeros_like(points)
    for i in range(3):  # x, y, z coordinates
        smoothed_points[:, i] = np.linalg.solve(A, points[:, i])
    
    return smoothed_points

def reconstruct_3d_trajectory(side_coordinates, back_coordinates, dlt_params, frame_rate):
    """
    Reconstruct 3D trajectory from 2D coordinates
    
    Parameters:
    - side_coordinates: List of 2D coordinates from side camera
    - back_coordinates: List of 2D coordinates from back camera
    - dlt_params: Dictionary containing DLT parameters for both cameras
    - frame_rate: Frame rate of the trajectory
    
    Returns:
    - Dictionary containing 3D points and analysis results
    """
    # Extract coordinates
    side_points = np.array([[c['x'], c['y']] for c in side_coordinates])
    back_points = np.array([[c['x'], c['y']] for c in back_coordinates])
    
    # Reconstruct 3D points
    points_3d = triangulate_points(back_points, side_points, 
                                  dlt_params['back'], dlt_params['side']) # Note: Triangulate points itself does not denormalize
    
    # --- Debugging: Check points_3d after triangulation ---
    print("DEBUG: Type of points_3d after triangulation:", type(points_3d))
    print("DEBUG: Shape of points_3d after triangulation:", points_3d.shape)
    print("DEBUG: Content of points_3d after triangulation:", points_3d)
    print("-------------------------------------------------")
    # ------------------------------------------------------
    
    # Convert to numpy array if not already
    points_3d = np.array(points_3d)
    
    # First detect and handle outliers
    valid_points_mask = detect_outliers(points_3d)
    if not np.all(valid_points_mask):
        print(f"Detected {np.sum(~valid_points_mask)} outliers in 3D reconstruction")
        # Replace outliers with interpolated values
        for i in range(3):  # x, y, z coordinates
            points_3d[~valid_points_mask, i] = np.interp(
                np.where(~valid_points_mask)[0],
                np.where(valid_points_mask)[0],
                points_3d[valid_points_mask, i]
            )
    
    # Apply cubic spline smoothing
    smoothed_points = smooth_3d_points(points_3d)
    
    # Convert back to list format for compatibility
    smoothed_points_list = smoothed_points.tolist()
    
    # Estimate throw distance and analyze trajectory
    distance_results = estimate_throw_distance(smoothed_points, frame_rate)
    
    # --- Debugging: Check distance_results before calling analyze_trajectory ---
    print("DEBUG: Type of distance_results before analyze_trajectory:", type(distance_results))
    print("DEBUG: Content of distance_results before analyze_trajectory:", distance_results)
    print("------------------------------------------------------------")
    # ---------------------------------------------------------------------
    
    analysis_results = analyze_trajectory(distance_results)
    
    return {
        'points_3d': smoothed_points_list,
        'distance_estimation': distance_results,
        'trajectory_analysis': analysis_results,
        'frame_rate': frame_rate
    } 