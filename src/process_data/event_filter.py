import numpy as np
from cv2 import pointPolygonTest
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch

def find_frame_indices_and_weights(event_ts, timestamps):
    """Find frame indices and interpolation weights for event timestamp"""
    idx = np.searchsorted(timestamps, event_ts)
    
    # Boundary handling
    if idx == 0:
        return 0, 0, 0.0  # Use first frame
    elif idx >= len(timestamps):
        return len(timestamps)-1, len(timestamps)-1, 0.0  # Use last frame
    
    # Normal case: event between two frames
    idx_prev, idx_next = idx-1, idx
    
    # Calculate interpolation weight (0-1)
    t_prev = timestamps[idx_prev]
    t_next = timestamps[idx_next]
    
    # Prevent division by zero
    alpha = 0.0 if t_next == t_prev else (event_ts - t_prev) / (t_next - t_prev)
    
    return idx_prev, idx_next, alpha


def interpolate_polygon(polygon_prev, polygon_next, alpha):
    """Linear interpolation of polygon vertex coordinates"""
    return (1 - alpha) * polygon_prev + alpha * polygon_next


def interpolate_angle(angle_prev, angle_next, alpha):
    """Linear interpolation of rotation angle"""
    # Handle angle crossing 360°
    diff = angle_next - angle_prev
    if diff > 180:
        angle_next -= 360
    elif diff < -180:
        angle_next += 360
    
    return (1 - alpha) * angle_prev + alpha * angle_next


def interpolate_center(center_prev, center_next, alpha):
    """Linear interpolation of center point coordinates"""
    return (1 - alpha) * center_prev + alpha * center_next


def get_frame_info(idx, crops_info):
    """Get frame information at specified index"""
    frame_info = crops_info[idx]
    if frame_info[0] is None:  # barbara_info is None
        return None, None, None, None
    
    barbara_info = frame_info[0]
    polygon = barbara_info['polygon'] if 'polygon' in barbara_info else None
    angle = barbara_info['angle'] if 'angle' in barbara_info else None
    center = barbara_info['center'] if 'center' in barbara_info else None
    
    return polygon, angle, center, frame_info[2]  # polygon, angle, center, timestamp


def get_interpolated_frame_info(event_ts, crops_info):
    """Get interpolated frame information for event"""
    # Extract all timestamps
    timestamps = np.array([info[2] for info in crops_info if info[2] is not None])
    
    # Find corresponding previous and next frames
    idx_prev, idx_next, alpha = find_frame_indices_and_weights(event_ts, timestamps)
    
    # Get previous and next frame information
    poly_prev, angle_prev, center_prev, _ = get_frame_info(idx_prev, crops_info)
    
    # If previous and next frames are the same, return frame information directly
    if idx_prev == idx_next:
        return poly_prev, angle_prev, center_prev
    
    poly_next, angle_next, center_next, _ = get_frame_info(idx_next, crops_info)
    
    # If one frame information is missing, use the other frame
    if poly_prev is None:
        return poly_next, angle_next, center_next
    if poly_next is None:
        return poly_prev, angle_prev, center_prev
    
    # Interpolation calculation
    interpolated_poly = interpolate_polygon(poly_prev, poly_next, alpha)
    interpolated_angle = interpolate_angle(angle_prev, angle_next, alpha)
    interpolated_center = interpolate_center(center_prev, center_next, alpha)
    
    return interpolated_poly, interpolated_angle, interpolated_center


def rotate_point(point, center, angle_deg):
    """Rotate point around center by specified angle
    
    Args:
        point: Point coordinates (x, y)
        center: Rotation center (cx, cy)
        angle_deg: Rotation angle (degrees)
    
    Returns:
        Rotated point coordinates
    """
    # Ensure numpy arrays
    if isinstance(point, torch.Tensor):
        point = point.cpu().numpy()
    if isinstance(center, torch.Tensor):
        center = center.cpu().numpy()
    
    # Convert to radians
    angle_rad = np.radians(angle_deg)
    
    # Translate to origin
    x, y = point
    cx, cy = center
    x_shifted, y_shifted = x - cx, y - cy
    
    # Rotate
    x_rotated = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
    y_rotated = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)
    
    # Translate back to original position
    return x_rotated + cx, y_rotated + cy


def adjust_polygon_to_target_size(polygon, center, angle, target_size):
    """Adjust polygon size according to target size, keeping center and rotation angle unchanged
    
    Args:
        polygon: Original polygon coordinates
        center: Center point
        angle: Rotation angle
        target_size: Target side length
    
    Returns:
        Adjusted polygon coordinates
    """
    # Calculate original dimensions
    width = np.linalg.norm(polygon[1] - polygon[0])
    height = np.linalg.norm(polygon[2] - polygon[1])
    
    # Use maximum side as reference
    max_size = max(width, height)
    
    # If target size not specified, use original size
    if target_size is None or target_size <= 0:
        return polygon
    
    # Calculate scale factor
    scale = target_size / max_size
    
    # Scale polygon (keeping center point unchanged)
    adjusted_polygon = np.zeros_like(polygon)
    for i, point in enumerate(polygon):
        # Vector from center to vertex
        vector = point - center
        # Scale vector
        scaled_vector = vector * scale
        # New vertex = center + scaled vector
        adjusted_polygon[i] = center + scaled_vector
    
    return adjusted_polygon


def point_in_rotated_box(point, polygon, center, angle):
    """Check if point is inside rotated crop box
    
    Args:
        point: Point coordinates (x, y)
        polygon: Polygon vertices
        center: Rotation center
        angle: Rotation angle (degrees)
    
    Returns:
        Whether point is inside box
    """
    # Reverse rotate point to make box horizontal
    x, y = rotate_point(point, center, -angle)
    
    # Ensure point coordinates are float type
    point = (float(x), float(y))
    
    # Ensure polygon is float32 type with correct shape
    polygon = polygon.astype(np.float32).reshape(-1, 1, 2)
    
    # Use OpenCV pointPolygonTest to check if point is inside polygon
    # Return value > 0 means point inside polygon, = 0 means on boundary, < 0 means outside
    return pointPolygonTest(polygon, point, False) >= 0


def transform_event_point(point, polygon, center, angle, output_size):
    """Transform event point coordinates to standardized output box
    
    Args:
        point: Event point coordinates (x, y)
        polygon: Polygon vertices
        center: Rotation center
        angle: Rotation angle (degrees)
        output_size: Output box size
    
    Returns:
        transformed_point: Transformed coordinates (x, y)
    """
    # Ensure center is numpy array
    if isinstance(center, torch.Tensor):
        center = center.cpu().numpy()
    
    # Ensure point is numpy array
    if isinstance(point, torch.Tensor):
        point = point.cpu().numpy()
    
    # Calculate position relative to center
    relative_point = np.array(point) - center
    
    # Calculate rotation matrix to cancel angle
    rot_rad = np.radians(-angle)
    cos_theta = np.cos(rot_rad)
    sin_theta = np.sin(rot_rad)
    rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    
    # Rotate point
    rotated_point = np.dot(rot_matrix, relative_point)
    
    # Translate to output box center
    output_center = np.array([output_size / 2, output_size / 2])
    transformed_point = rotated_point + output_center
    
    # Clip coordinates and ensure integers
    transformed_point = np.clip(transformed_point, 0, output_size - 1)
    transformed_point = np.floor(transformed_point).astype(int)
    
    return transformed_point


def filter_event(event, crops_info, target_size=None, transform=True):
    """Process single event, determine if valid, considering box rotation
    
    Args:
        event: Single event data, format [timestamp, x, y, polarity]
        crops_info: Crop box information list
        target_size: Target crop box size, use original size if None
        transform: Whether to transform event coordinates
        
    Returns:
        If valid event and transform=False, return original event
        If valid event and transform=True, return transformed event
        Otherwise return None
    """
    # Extract event information, ensure numpy values
    if isinstance(event, torch.Tensor):
        timestamp = event[0].item()
        x = event[1].item()
        y = event[2].item()
        polarity = event[3].item()
    else:
        timestamp, x, y, polarity = event
    
    # Get interpolated crop box information
    polygon, angle, center = get_interpolated_frame_info(timestamp, crops_info)
    
    # If unable to get valid crop box information, return None
    if polygon is None or angle is None or center is None:
        return None
    
    # Adjust polygon according to target size
    if target_size is not None:
        polygon = adjust_polygon_to_target_size(polygon, center, angle, target_size)
    
    # Check if event is inside rotated crop box
    if point_in_rotated_box((x, y), polygon, center, angle):
        if transform:
            # Transform event coordinates
            output_size = target_size if target_size is not None else max(
                np.linalg.norm(polygon[1] - polygon[0]),
                np.linalg.norm(polygon[2] - polygon[1])
            )
            new_x, new_y = transform_event_point((x, y), polygon, center, angle, output_size)
            return [timestamp, new_x, new_y, polarity]
        else:
            # Return original event
            return event.tolist() if isinstance(event, torch.Tensor) else event
    
    return None


def preprocess_crops_info(crops_info):
    """Keep only frames with valid polygon/angle/center and timestamp"""
    polygons = []
    angles = []
    centers = []
    timestamps = []

    for info in crops_info:
        # Check if barbara_info (info[0]) and timestamp (info[2]) exist
        if info[0] is None or info[2] is None:
            continue
            
        barbara = info[0]
        # Check if geometric information is complete and valid
        if ('polygon' not in barbara or
            'angle' not in barbara or
            'center' not in barbara or
            barbara['polygon'] is None or
            barbara['angle'] is None or
            barbara['center'] is None):
            continue

        # Ensure polygon is 4x2
        current_polygon = np.asarray(barbara['polygon'], dtype=np.float32)
        if current_polygon.shape != (4, 2):
             continue
             
        polygons.append(current_polygon)
        angles.append(barbara['angle'])
        centers.append(np.asarray(barbara['center'], dtype=np.float32))
        timestamps.append(info[2])

    # Check if valid data collected
    if not timestamps:
        print("Warning: preprocess_crops_info found no frames with valid geometric information and timestamps.")
        return np.empty((0, 4, 2)), np.empty((0,)), np.empty((0, 2)), np.empty((0,))

    # Convert to numpy
    try:
        polygons_np = np.stack(polygons, axis=0)  # [M_valid, 4, 2]
        angles_np = np.asarray(angles, dtype=np.float32)
        centers_np = np.stack(centers, axis=0)    # [M_valid, 2]
        timestamps_np = np.asarray(timestamps, dtype=np.float64)
    except ValueError as e:
        print(f"Error: Failed to convert list to NumPy array: {e}")
        return np.empty((0, 4, 2)), np.empty((0,)), np.empty((0, 2)), np.empty((0,))
        
    # Ensure timestamps are sorted, crucial for np.searchsorted
    sort_indices = np.argsort(timestamps_np)
    
    return polygons_np[sort_indices], angles_np[sort_indices], centers_np[sort_indices], timestamps_np[sort_indices]


def filter_events_all(events_tensor, crops_info, target_size=None, transform=False):
    """
    Fully vectorized event data filtering.
    This function efficiently processes large numbers of events through NumPy vectorized operations.
    Core steps:
    1. Preprocess crop box information, filter valid frames and sort.
    2. Find corresponding previous/next frame indices and interpolation weights for each event.
    3. Get or interpolate polygon, rotation angle, and center point for each event.
    4. (Optional) Adjust polygon size according to target_size.
    5. Reverse rotate event points relative to their corresponding polygon rotation.
    6. Use cross product method to check if reverse-rotated event points are inside adjusted polygons.
    7. (Optional) If transform is True, transform valid event coordinates to normalized target box.
    """
    if not isinstance(events_tensor, np.ndarray):
        if isinstance(events_tensor, torch.Tensor):
            events = events_tensor.cpu().numpy()
        else:
            try:
                events = np.asarray(events_tensor)
            except Exception as e:
                print(f"Error: Input events_tensor type cannot be directly converted to NumPy array: {type(events_tensor)}, error: {e}")
                return np.empty((0, 4))
    else:
        events = events_tensor

    if events.ndim != 2 or events.shape[1] != 4:
        print(f"Error: events_tensor shape should be [N, 4], actual: {events.shape}")
        return np.empty((0,4))

    if len(events) == 0:
        return np.empty((0, 4))

    # 1. Preprocess crops_info: get valid frame polygons, angles, centers, timestamps, sorted by timestamp
    polygons_data, angles_data, centers_data, frame_ts = preprocess_crops_info(crops_info)
    n_frames = len(frame_ts)
    if n_frames == 0:
        return np.empty((0, 4))

    # 2. Extract basic event data
    timestamps = events[:, 0]
    points = events[:, 1:3]  # Original event (x, y) coordinates
    polarities = events[:, 3]
    n_events = len(events)

    # 3. Find index for each event timestamp in sorted frame timestamps
    idx = np.searchsorted(frame_ts, timestamps, side='right')
    idx_prev = np.clip(idx - 1, 0, n_frames - 1)
    idx_next = np.clip(idx, 0, n_frames - 1)

    # 4. Calculate interpolation weight alpha
    alpha = np.zeros(n_events, dtype=float)
    needs_interp_mask = (idx > 0) & (idx < n_frames)
    if np.any(needs_interp_mask):
        _t_events = timestamps[needs_interp_mask]
        _t_prev = frame_ts[idx_prev[needs_interp_mask]]
        _t_next = frame_ts[idx_next[needs_interp_mask]]
        
        denominator = _t_next - _t_prev
        calc_alpha_mask = denominator != 0
        if np.any(calc_alpha_mask):
             alpha_update_indices = np.where(needs_interp_mask)[0][calc_alpha_mask]
             alpha[alpha_update_indices] = (_t_events[calc_alpha_mask] - _t_prev[calc_alpha_mask]) / denominator[calc_alpha_mask]
             alpha[alpha_update_indices] = np.clip(alpha[alpha_update_indices], 0.0, 1.0)

    # 5. Get or interpolate polygon, angle, center for each event
    interp_polygons = np.copy(polygons_data[idx_prev])
    interp_angles = np.copy(angles_data[idx_prev])
    interp_centers = np.copy(centers_data[idx_prev])
    
    interp_update_mask = alpha > 0
    if np.any(interp_update_mask):
        interp_indices = np.where(interp_update_mask)[0]
        _alpha_val = alpha[interp_indices]
        _idx_prev_val = idx_prev[interp_indices]
        _idx_next_val = idx_next[interp_indices]

        weights = _alpha_val.reshape(-1, 1, 1)
        interp_polygons[interp_indices] = (1 - weights) * polygons_data[_idx_prev_val] + weights * polygons_data[_idx_next_val]
        
        # Angle interpolation, handle circular crossing
        angle_diff = (angles_data[_idx_next_val] - angles_data[_idx_prev_val] + 180) % 360 - 180
        interp_angles[interp_indices] = angles_data[_idx_prev_val] + angle_diff * _alpha_val
        
        interp_centers[interp_indices] = (
            (1 - _alpha_val).reshape(-1, 1) * centers_data[_idx_prev_val] +
            _alpha_val.reshape(-1, 1) * centers_data[_idx_next_val]
        )

    # 6. (Optional) Adjust polygon size according to target_size
    adjusted_polygons = interp_polygons
    if target_size is not None and target_size > 0:
        _polygons_to_adjust = interp_polygons
        _centers_for_adjust = interp_centers

        rel_vectors = _polygons_to_adjust - _centers_for_adjust[:, None, :]
        width = np.linalg.norm(_polygons_to_adjust[:, 1] - _polygons_to_adjust[:, 0], axis=1)
        height = np.linalg.norm(_polygons_to_adjust[:, 2] - _polygons_to_adjust[:, 1], axis=1)
        current_max_size = np.maximum(width, height)
        
        scale_factors = np.zeros_like(current_max_size)
        valid_scale_mask = current_max_size > 0
        scale_factors[valid_scale_mask] = target_size / current_max_size[valid_scale_mask]
        
        rel_vectors_scaled = rel_vectors * scale_factors[:, None, None]
        adjusted_polygons = rel_vectors_scaled + _centers_for_adjust[:, None, :]

    # 7. Prepare event point coordinates for "point in polygon" test
    points_centered_for_test = points - interp_centers
    angles_rad_for_test = np.radians(-interp_angles)
    
    cos_theta_test = np.cos(angles_rad_for_test)
    sin_theta_test = np.sin(angles_rad_for_test)
    
    rotated_x_test = points_centered_for_test[:, 0] * cos_theta_test - points_centered_for_test[:, 1] * sin_theta_test
    rotated_y_test = points_centered_for_test[:, 0] * sin_theta_test + points_centered_for_test[:, 1] * cos_theta_test
    
    points_for_test = np.column_stack((rotated_x_test, rotated_y_test)) + interp_centers

    # 8. Use cross product method to check if points_for_test are inside adjusted_polygons
    poly_a = adjusted_polygons[:, 0]; poly_b = adjusted_polygons[:, 1]
    poly_c = adjusted_polygons[:, 2]; poly_d = adjusted_polygons[:, 3]
    
    cross_ab = np.cross(poly_b - poly_a, points_for_test - poly_a)
    cross_bc = np.cross(poly_c - poly_b, points_for_test - poly_b)
    cross_cd = np.cross(poly_d - poly_c, points_for_test - poly_c)
    cross_da = np.cross(poly_a - poly_d, points_for_test - poly_d)
    
    epsilon = 1e-6

    in_polygon_ccw = (
        (cross_ab >= -epsilon) & (cross_bc >= -epsilon) &
        (cross_cd >= -epsilon) & (cross_da >= -epsilon)
    )
    in_polygon_cw = (
        (cross_ab <= epsilon) & (cross_bc <= epsilon) &
        (cross_cd <= epsilon) & (cross_da <= epsilon)
    )
    in_polygon_mask = in_polygon_ccw | in_polygon_cw
            
    if not np.any(in_polygon_mask):
        return np.empty((0, 4))

    # 9. (Optional) If transform is True, transform valid event coordinates to target output box
    if transform:
        valid_event_indices = np.where(in_polygon_mask)[0]
        
        _points_to_transform = points[valid_event_indices]
        _centers_for_transform = interp_centers[valid_event_indices]
        _angles_for_transform = interp_angles[valid_event_indices]
        
        if target_size is not None and target_size > 0:
            output_size = target_size
        else:
            valid_output_polygons = adjusted_polygons[in_polygon_mask]
            if len(valid_output_polygons) == 0:
                output_size = 0
            else:
                output_size = np.max([
                    np.linalg.norm(valid_output_polygons[:, 1] - valid_output_polygons[:, 0], axis=1),
                    np.linalg.norm(valid_output_polygons[:, 2] - valid_output_polygons[:, 1], axis=1)
                ])

        if output_size <= 0:
            return np.empty((0,4))

        points_centered_for_transform = _points_to_transform - _centers_for_transform
        angles_rad_for_transform = np.radians(-_angles_for_transform)
        
        cos_theta_transform = np.cos(angles_rad_for_transform)
        sin_theta_transform = np.sin(angles_rad_for_transform)
        
        rot_matrices = np.zeros((len(valid_event_indices), 2, 2))
        rot_matrices[:, 0, 0] = cos_theta_transform
        rot_matrices[:, 0, 1] = -sin_theta_transform
        rot_matrices[:, 1, 0] = sin_theta_transform
        rot_matrices[:, 1, 1] = cos_theta_transform
        
        points_centered_exp = points_centered_for_transform.reshape(-1, 2, 1)
        rotated_points_for_output = np.matmul(rot_matrices, points_centered_exp).squeeze(axis=2)
        
        output_frame_center = np.array([output_size / 2, output_size / 2])
        transformed_coords = rotated_points_for_output + output_frame_center
        
        transformed_coords = np.clip(transformed_coords, 0, output_size - 1)
        final_transformed_coords = np.floor(transformed_coords).astype(int)
        
        transformed_points_output_all = np.zeros((n_events, 2), dtype=int)
        transformed_points_output_all[valid_event_indices] = final_transformed_coords
        
        result_events = np.column_stack((
            timestamps[in_polygon_mask],
            transformed_points_output_all[in_polygon_mask],
            polarities[in_polygon_mask]
        ))
    else:
        result_events = events[in_polygon_mask]

    return result_events


def _filter_batch(args):
    """Process single batch of event data (for multi-threading)
    
    Args:
        args: Tuple (batch_events, crops_info, target_size, transform)
    
    Returns:
        Filtered event data
    """
    batch_events, crops_info, target_size, transform = args
    return filter_events_all(batch_events, crops_info, target_size, transform)

def filter_events_parallel(events_tensor, crops_info, target_size=None, transform=False, 
                          batch_size=100000, n_workers=4):
    """Parallel process event data using batch + multi-threading
    
    Args:
        events_tensor: Event data tensor [N, 4] (timestamp, x, y, polarity)
        crops_info: Crop box information list
        target_size: Target crop box size
        transform: Whether to transform event coordinates
        batch_size: Number of events per batch
        n_workers: Number of parallel threads
    
    Returns:
        filtered_events: Time-sorted filtered event data
    """
    # Handle empty input
    if len(events_tensor) == 0:
        return np.empty((0, 4))
    
    # Split into batches
    batches = []
    for i in range(0, len(events_tensor), batch_size):
        batch_events = events_tensor[i:i+batch_size]
        batches.append((batch_events, crops_info, target_size, transform))
    
    # Use multi-threading for parallel processing with progress bar
    results = []
    batch_valid_counts = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_filter_batch, batch) for batch in batches]
        
        for future in tqdm(
            futures, 
            total=len(batches),
            desc="Processing event batches",
            unit="batch"
        ):
            result = future.result()
            results.append(result)
            batch_valid_counts.append(len(result))
        
    print(f"Total batches: {len(batch_valid_counts)}, batches with valid events: {sum(1 for c in batch_valid_counts if c > 0)}")
    
    # Merge results
    valid_results = [result for result in results if len(result) > 0]
    if valid_results:
        merged_events = np.vstack(valid_results)
        sorted_indices = np.argsort(merged_events[:, 0])
        return merged_events[sorted_indices]
    else:
        return np.empty((0, 4))