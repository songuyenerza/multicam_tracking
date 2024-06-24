import numpy as np
from scipy.optimize import linear_sum_assignment


def is_point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def match_points_across_frames(dict_points_all_frame, threshold=100):
    thread_names = list(dict_points_all_frame.keys())
    if len(thread_names) < 2:
        return [(tuple(point),) for points in dict_points_all_frame.values() for point in points]

    all_points = [np.array(dict_points_all_frame[thread_name]) for thread_name in thread_names]
    num_frames = len(all_points)
    
    matched_points = []
    all_matched_indices = [set() for _ in range(num_frames)]
    
    for i in range(num_frames - 1):
        for j in range(i + 1, num_frames):
            points1 = all_points[i]
            points2 = all_points[j]
            
            # Compute distance matrix
            # dist_matrix = np.linalg.norm(points1[:, np.newaxis] - points2, axis=2)
            dist_matrix = np.abs(points1[:, np.newaxis, 0] - points2[:, 0])

            max_value = np.max(dist_matrix)
            dist_matrix[dist_matrix > threshold] = max_value

            # Apply Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            # Filter matches based on a distance threshold
            for r, c in zip(row_ind, col_ind):
                if dist_matrix[r, c] < threshold:
                    matched_points.append((tuple(points1[r]), tuple(points2[c])))
                    all_matched_indices[i].add(r)
                    all_matched_indices[j].add(c)

    # Add unmatched points
    for k in range(num_frames):
        unmatched_points = [tuple(point) for idx, point in enumerate(all_points[k]) if idx not in all_matched_indices[k]]
        matched_points.extend([(point,) for point in unmatched_points])
    # print("matched_points: ", matched_points)
    return matched_points