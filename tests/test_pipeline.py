import os
import sys
import numpy as np

# Add parent directory to path so that severity_estimator can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from severity_estimator import estimate_pothole_metrics, calculate_road_health_score

def test_estimate_pothole_metrics_basic():
    # Create a dummy disparity map of shape (100, 100)
    # Background road is 100.
    disparity_map = np.ones((100, 100), dtype=np.uint8) * 100
    
    # Create a pothole at bbox [10, 10, 20, 20] (W=10, H=10, Area=100)
    # The bottom is at 60 (disparity difference of 40)
    disparity_map[10:20, 10:20] = 80
    disparity_map[13:17, 13:17] = 60 # center bottom is 60
    
    bbox = [10, 10, 20, 20]
    metrics = estimate_pothole_metrics(bbox, disparity_map)
    
    assert metrics["width"] == 10
    assert metrics["height"] == 10
    assert metrics["bbox_area"] == 100
    
    # Inside the box is 80s and 60s. The 5th percentile will be 60.
    # The border pixels will be 80 (since we index box_disp[ymin:ymax, xmin:xmax] which is 80)
    # Let's inspect the math:
    # boundary of box_disp: top edge is index 0 of box_disp, which is slice [10, 10:20] which is 80.
    # So road disparity should be ~80.
    # Bottom disparity (5th percentile) is 60.
    # Depth score ~ 80 - 60 = 20.
    assert metrics["bottom_disparity"] == 60.0
    assert metrics["depth_score"] > 0
    assert metrics["severity"] in ["Mild", "Moderate", "Severe"]

def test_road_health_score():
    # Empty detections -> 100
    assert calculate_road_health_score([]) == 100
    
    # Mild deduction
    assert calculate_road_health_score(["Mild"]) == 95
    
    # Moderate deduction
    assert calculate_road_health_score(["Moderate"]) == 90
    
    # Severe deduction
    assert calculate_road_health_score(["Severe"]) == 80
    
    # Mixed deductions
    assert calculate_road_health_score(["Mild", "Moderate", "Severe"]) == 65
    
    # Floor check
    large_list = ["Severe"] * 6 # 6 * 20 = 120 deduction
    assert calculate_road_health_score(large_list) == 0

if __name__ == "__main__":
    test_estimate_pothole_metrics_basic()
    test_road_health_score()
    print("All unit tests passed successfully!")
