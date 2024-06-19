import cv2
import numpy as np
import os


folder_frame = "/home/sonlt373/Desktop/SoNg/Multi_cam_tracking/dev/multicam_tracking/sample_data/frame_video_1"
# Define the path for the image and the points
img_path = os.path.join(folder_frame, "frame.jpg")
point_path = os.path.join(folder_frame, "point.txt")

# Read the image
image = cv2.imread(img_path)
height, width = image.shape[:2]

# Read the normalized points from the file
with open(point_path, 'r') as file:
    points_str = file.read().strip().split(';')
    points = [float(p) for p in points_str]


# Denormalize the points
denormalized_points = [
    [points[0] * width, points[1] * height],
    [points[2] * width, points[3] * height],
    [points[4] * width, points[5] * height],
    [points[6] * width, points[7] * height]
]
denormalized_points = np.array(denormalized_points, dtype=np.float32)

height = max(
    np.linalg.norm(denormalized_points[0] - denormalized_points[1]),
    np.linalg.norm(denormalized_points[3] - denormalized_points[2])
)
width = max(
    np.linalg.norm(denormalized_points[0] - denormalized_points[3]),
    np.linalg.norm(denormalized_points[1] - denormalized_points[2])
)

width = int(width / (height /300))
height = 300
print("width: ", width, " height: ", height)

# Define the destination points for the warp (e.g., rectangle)
dst_points = np.array([
    [width + 2146, height],  # Bottom right
    [width + 2146, 0],           # Top right
    [2146, 0],                   # Top left
    [2146, height]           # Bottom left
], dtype=np.float32)

# dst_points = np.array([
#     [width, height],  # Bottom right
#     [width, 0],           # Top right
#     [0, 0],                   # Top left
#     [0, height]           # Bottom left
# ], dtype=np.float32)
print("denormalized_points: ", denormalized_points)

H, _ = cv2.findHomography(denormalized_points, dst_points)
# Calculate the perspective transform matrix
matrix = cv2.getPerspectiveTransform(denormalized_points, dst_points)
print("matrix: ", matrix)
print("matrix: ", H)

matrix_array = np.array(H)

# Lưu ma trận vào tệp tin văn bản
np.savetxt(os.path.join(folder_frame,'matrix_homo.txt'), matrix_array)

# Warp the image
warped_image = cv2.warpPerspective(image, H, ((int(width + 2169), int(height))))
cv2.line(warped_image, (2146, 0), (2146, 300), (0, 0, 255), thickness=2)


# # Save and show the warped image
output_path = os.path.join(folder_frame, "warped_frame.jpg")
cv2.imwrite(output_path, warped_image)
cv2.imshow("Warped Image", warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
