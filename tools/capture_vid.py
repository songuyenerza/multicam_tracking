import cv2
import os
from tqdm import tqdm

# Path to the folder containing videos
video_path = "/home/sonlt373/Desktop/SoNg/Multi_cam_tracking/dev/multicam_tracking/sample_data/x5sonnt.mp4"

# Output folder to save frames
output_folder = "/home/sonlt373/Desktop/SoNg/Multi_cam_tracking/dev/multicam_tracking/sample_data/frame_video_2"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

    
# Check if the path is a file (to skip any subdirectories)
if os.path.isfile(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the base name of the video file without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save the frame as an image file
        frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count}.jpg")
        # if frame_count % 300 == 0:
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
        break
    
    # Release the video capture object
    cap.release()

print("Frame extraction completed.")