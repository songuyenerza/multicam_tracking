import cv2
import numpy as np

# Define the path for the image and the points
img_path = "/Users/sonnguyen/Desktop/AI/GHTK/multicam_tracking/sample_data/frame_video_2/frame2.jpg"
point_path = "/Users/sonnguyen/Desktop/AI/GHTK/multicam_tracking/sample_data/frame_video_2/point2.txt"

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

# Convert to numpy array

# Define the destination points for the warp (e.g., rectangle)
dst_points = np.array([
    [width - 1, height - 1],  # Bottom right
    [width - 1, 0],           # Top right
    [0, 0],                   # Top left
    [0, height - 1]           # Bottom left
], dtype=np.float32)

# Calculate the perspective transform matrix
matrix = cv2.getPerspectiveTransform(denormalized_points, dst_points)
print("matrix: ", matrix)

# exit()

# Warp the image
warped_image = cv2.warpPerspective(image, matrix, ((int(width), int(height))))

# # Save and show the warped image
output_path = "/Users/sonnguyen/Desktop/AI/GHTK/multicam_tracking/sample_data/frame_video_2/warped_frame2.jpg"
cv2.imwrite(output_path, warped_image)
# cv2.imshow("Warped Image", warped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
