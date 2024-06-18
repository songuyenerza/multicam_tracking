import cv2
import numpy as np

# Define the path for the image and the points
img_path = "/home/sonlt373/Desktop/SoNg/Multi_cam_tracking/dev/multicam_tracking/sample_data/frame_video_2_/frame2.jpg"
point_path = "/home/sonlt373/Desktop/SoNg/Multi_cam_tracking/dev/multicam_tracking/sample_data/frame_video_2_/point2.txt"

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

# Convert to numpy array
denormalized_points = np.array(denormalized_points, dtype=np.float32)

# Define the destination points for the warp (e.g., rectangle)
dst_points = np.array([
    [width - 1, height - 1],  # Bottom right
    [width - 1, 0],           # Top right
    [0, 0],                   # Top left
    [0, height - 1]           # Bottom left
], dtype=np.float32)

# Calculate the perspective transform matrix
matrix = cv2.getPerspectiveTransform(denormalized_points, dst_points)

# Warp the image
warped_image = cv2.warpPerspective(image, matrix, (width, height))

# # Save and show the warped image
# output_path = "/home/sonlt373/Desktop/SoNg/Multi_cam_tracking/dev/multicam_tracking/sample_data/frame_video_2_/warped_frame2.jpg"
# cv2.imwrite(output_path, warped_image)
cv2.imshow("Warped Image", warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
