import numpy as np

def estimate_pothole_metrics(bbox, disparity_map, area_thresholds=None, depth_thresholds=None):
    """
    Estimate pothole metrics: area, depth (disparity difference), and severity.
    
    Parameters:
    - bbox: list or tuple of [xmin, ymin, xmax, ymax]
    - disparity_map: 2D numpy array (grayscale) representing disparity values
    - area_thresholds: tuple (mild_max, mod_max) for area in pixels
    - depth_thresholds: tuple (mild_max, mod_max) for depth in disparity difference units
    
    Returns:
    - dict with: width, height, bbox_area, road_disparity, bottom_disparity, depth_score, severity
    """
    if area_thresholds is None:
        area_thresholds = (1200, 3500)
    if depth_thresholds is None:
        depth_thresholds = (12.0, 28.0)
        
    xmin, ymin, xmax, ymax = map(int, bbox)
    H, W = disparity_map.shape
    
    # Ensure coordinates are within image boundaries
    xmin = max(0, min(xmin, W - 1))
    xmax = max(0, min(xmax, W - 1))
    ymin = max(0, min(ymin, H - 1))
    ymax = max(0, min(ymax, H - 1))
    
    # Bbox size
    width = xmax - xmin
    height = ymax - ymin
    bbox_area = width * height
    
    if bbox_area == 0:
        return {
            "width": 0, "height": 0, "bbox_area": 0,
            "road_disparity": 0.0, "bottom_disparity": 0.0,
            "depth_score": 0.0, "severity": "Mild"
        }
        
    # Extract bounding box region from disparity map
    box_disp = disparity_map[ymin:ymax, xmin:xmax]
    
    # 1. Road Surface Disparity: average of boundary pixels of the bounding box
    boundary_pixels = []
    # top and bottom edges
    boundary_pixels.extend(box_disp[0, :])
    if box_disp.shape[0] > 1:
        boundary_pixels.extend(box_disp[-1, :])
    # left and right edges (excluding corners to avoid redundancy)
    if box_disp.shape[0] > 2:
        boundary_pixels.extend(box_disp[1:-1, 0])
        if box_disp.shape[1] > 1:
            boundary_pixels.extend(box_disp[1:-1, -1])
            
    road_disparity = np.mean(boundary_pixels) if boundary_pixels else float(np.mean(box_disp))
    
    # 2. Pothole Bottom Disparity: 5th percentile to be robust against noise
    bottom_disparity = np.percentile(box_disp, 5)
    
    # 3. Depth score = road_disparity - bottom_disparity (larger depth = lower disparity)
    depth_score = max(0.0, road_disparity - bottom_disparity)
    
    # 4. Severity classification based on area and depth
    # We use a combined logic: if either exceeds the moderate/severe threshold, classify accordingly
    if bbox_area >= area_thresholds[1] or depth_score >= depth_thresholds[1]:
        severity = "Severe"
    elif bbox_area >= area_thresholds[0] or depth_score >= depth_thresholds[0]:
        severity = "Moderate"
    else:
        severity = "Mild"
        
    return {
        "width": width,
        "height": height,
        "bbox_area": bbox_area,
        "road_disparity": float(road_disparity),
        "bottom_disparity": float(bottom_disparity),
        "depth_score": float(depth_score),
        "severity": severity
    }

def calculate_road_health_score(severities):
    """
    Calculate Road Health Score (0-100) based on severity list of detected potholes.
    Deductions:
    - Mild: -5
    - Moderate: -10
    - Severe: -20
    """
    score = 100
    deductions = {
        "Mild": 5,
        "Moderate": 10,
        "Severe": 20
    }
    
    for sev in severities:
        score -= deductions.get(sev, 0)
        
    return max(0, min(score, 100))
