import numpy as np

def extract_surgical_features(hand_landmarks):
    # 1. Raw Coordinates (63 values)
    raw_coords = []
    pts = []
    for lm in hand_landmarks:
        raw_coords.extend([lm.x, lm.y, lm.z])
        pts.append(np.array([lm.x, lm.y, lm.z]))

    # 2. Scale-Invariant Feature Engineering:
    # Scale Ruler: Distance from Wrist (point 0) to Middle Finger Base Knuckle (point 9)
    wrist = pts[0]
    middle_mcp = pts[9]
    scale_ruler = np.linalg.norm(middle_mcp - wrist)
    if scale_ruler == 0:
        scale_ruler = 1e-6

    # Normalized distances from wrist to all 5 fingertips
    # Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20
    fingertip_ids = [4, 8, 12, 16, 20]
    norm_distances = [
        np.linalg.norm(pts[tip] - wrist) / scale_ruler
        for tip in fingertip_ids
    ]

    # Calculate angles between fingers
    def compute_angle(p1, vertex, p2):
        v1 = p1 - vertex
        v2 = p2 - vertex
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    # Angle 1: Thumb tip - Wrist - Index tip
    angle_thumb_index = compute_angle(pts[4], wrist, pts[8])
    # Angle 2: Index tip - Wrist - Middle tip
    angle_index_middle = compute_angle(pts[8], wrist, pts[12])

    invariant_features = norm_distances + [angle_thumb_index, angle_index_middle]

    return np.array(raw_coords), np.array(invariant_features)