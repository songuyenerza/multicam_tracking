import cv2
import numpy as np
import os

def read_points(point_path, width, height):
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
    return np.array(denormalized_points, dtype=np.float32)

def calculate_output_size(points):
    height = max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[3] - points[2])
    )
    width = max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2])
    )

    width = int(width / (height / 300))
    height = 300
    return width, height

def process_frame(folder_frame, reference_point):
    # Define the path for the image and the points
    img_path = os.path.join(folder_frame, "frame.jpg")
    point_path = os.path.join(folder_frame, "point.txt")

    # Read the image
    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    # Read and denormalize the points
    denormalized_points = read_points(point_path, width, height)

    # Calculate the output size
    out_width, out_height = calculate_output_size(denormalized_points)

    print("width: ", out_width, " height: ", out_height)

    # Define the destination points for the warp
    dst_points = np.array([
        [out_width + reference_point, out_height],  # Bottom right
        [out_width + reference_point, 0],           # Top right
        [reference_point, 0],                       # Top left
        [reference_point, out_height]               # Bottom left
    ], dtype=np.float32)

    print("denormalized_points: ", denormalized_points)

    # Calculate the homography matrix
    H, _ = cv2.findHomography(denormalized_points, dst_points)
    print("Homography Matrix: ", H)

    # Save the homography matrix
    matrix_array = np.array(H)
    np.savetxt(os.path.join(folder_frame, 'matrix_homo.txt'), matrix_array)

    # Warp the image
    warped_image = cv2.warpPerspective(image, H, (int(out_width + reference_point), int(out_height)))
    cv2.line(warped_image, (reference_point, 0), (reference_point, 300), (0, 0, 255), thickness=2)

    # Save and show the warped image
    output_path = os.path.join(folder_frame, "warped_frame.jpg")
    cv2.imwrite(output_path, warped_image)

    return warped_image

# Example usage
if __name__ == "__main__":
    folder_frame1 = "./sample_data/frame_video_1"
    folder_frame2 = "./sample_data/frame_video_2"
    
    # Process each frame folder
    warped_image1 = process_frame(folder_frame1, 0)
    warped_image2 = process_frame(folder_frame2, 0)

    # Combine the warped images into one overview image
    combined_height = max(warped_image1.shape[0], warped_image2.shape[0])
    combined_width = warped_image1.shape[1] + warped_image2.shape[1]

    overview_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    overview_image[:, :warped_image2.shape[1]] = warped_image2
    overview_image[:, warped_image2.shape[1]:] = warped_image1

    # Save the overview image
    output_overview_path = "./sample_data/overview_image.jpg"
    cv2.imwrite(output_overview_path, overview_image)

    # Display the overview image
    cv2.imshow("Overview Image", overview_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
