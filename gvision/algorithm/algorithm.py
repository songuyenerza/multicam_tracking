import numpy as np
from scipy.optimize import linear_sum_assignment
from gvision.constants import *


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


def Read_roi_file(roi_path, W, H):
    with open(roi_path, 'r') as file:
        roi_data = file.readline().strip().split(';')
        roi_data = [float(x) for x in roi_data]

        # Convert ROI data into pairs
        roi_pairs = [(roi_data[i] * W, roi_data[i + 1] * H) for i in range(0, len(roi_data), 2)]
        roi_pairs = [(int(x), int(y)) for x, y in roi_pairs]
    return roi_pairs


def Warp_box_to_overview(dict_bboxes_homo_all_frame, roi_pairs):
    # warp base on homography matrix
    dict_result = {}
    for name_stream in dict_bboxes_homo_all_frame.keys():
        box_xyxys = dict_bboxes_homo_all_frame[name_stream]["bbox"]
        homo_matrix = dict_bboxes_homo_all_frame[name_stream]["homo_matrix"]

        results = []
        for box_xyxy in box_xyxys:

            # Extract the coordinates of the bounding box corners
            x1, y1, x2, y2 = box_xyxy

            # Transform the top-left corner (x1, y1)
            top_left = np.dot(homo_matrix, np.array([x1, y1, 1]))
            top_left = top_left / top_left[2]
            transformed_top_left = top_left[:2]

            # Transform the bottom-right corner (x2, y2)
            bottom_right = np.dot(homo_matrix, np.array([x2, y2, 1]))
            bottom_right = bottom_right / bottom_right[2]
            transformed_bottom_right = bottom_right[:2]

            # Calculate the center of the bounding box in the warped image
            point_center_x = round((transformed_top_left[0] + transformed_bottom_right[0]) / 2, 2)
            point_center_y = round((transformed_top_left[1] + transformed_bottom_right[1]) / 2, 2)
            point_center = (point_center_x, point_center_y)

            if is_point_in_polygon(point_center, roi_pairs):
                if name_stream == "Stream_1":
                    if W1 + W2 + W3 + W4 + W5 + W6 <  point_center[0] <= WIDTH_IMAGE_OVERVIEW:
                        results.append(point_center)
                elif name_stream == "Stream_2":
                    if W1 + W2 + W3 + W4 + W5 - 200 <  point_center[0] <= WIDTH_IMAGE_OVERVIEW - W7 + 100:
                        results.append(point_center)
                elif name_stream == "Stream_3":
                    if W1 + W2 + W3 + W4 - 100 <  point_center[0] <= WIDTH_IMAGE_OVERVIEW - W7 - W6 + 150:
                        results.append(point_center)

                elif name_stream == "Stream_4":
                    if W1 + W2 + W3 <  point_center[0] <= WIDTH_IMAGE_OVERVIEW - W7 - W6 - W5:
                        results.append(point_center)
                elif name_stream == "Stream_5":
                    if W1 + W2  <  point_center[0] <= WIDTH_IMAGE_OVERVIEW - W7 - W6 - W5 - W4:
                        results.append(point_center)
                elif name_stream == "Stream_6":
                    if W1 <  point_center[0] <= WIDTH_IMAGE_OVERVIEW - W7 - W6 - W5 - W4 - W3:
                        results.append(point_center)
                elif name_stream == "Stream_7":
                    if 0 <  point_center[0] <= WIDTH_IMAGE_OVERVIEW - W7 - W6 - W5 - W4 - W3 - W2:
                        results.append(point_center)

        if len(results) > 0:
            dict_result[name_stream] = results

    return dict_result

