import threading
import cv2
from ultralytics import YOLO
from queue import Queue
import time
import numpy as np

# Initialize the YOLO model
model = YOLO('/home/sonlt373/Desktop/SoNg/Tracking/240124_yolov8s_package_640.pt')

# Define a function to process video frames
def process_video(video_path, result_queue, thread_name, homo_matrix):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if thread_name == "Thread_1" and frame_count < 14:
            frame_count += 1
            continue
        predictions = model.predict(source = frame,
                                imgsz = 640,
                                conf = 0.4,
                                iou = 0.6,
                                verbose = False,
                                save = True)
        print("---------------------------------------")    
        results = []
        
        for i, box in enumerate(predictions[0].boxes):
            cls = int(box.cls)
            box_xyxy = [float(num) for num in box.xyxy[0]]

            transformed_box = np.dot(homo_matrix, np.array([box_xyxy[0], box_xyxy[1], 1]).reshape((3, 1)))
            point_center = transformed_box.reshape(1, 3)[0][:2]

            point_center = (int(point_center[0]), int(point_center[1]))
            results.append(point_center)

        # Put the results in the queue
        result_queue.put((thread_name, frame, results))

    cap.release()
    result_queue.put((thread_name, None, None))

# Function to process results in the main thread
def process_results(result_queue):
    while True:
        try:
            image_bgr = np.ones((300, 4315, 3), dtype=np.uint8) * 128
            cv2.line(image_bgr, (2146, 0), (2146, 300), (0, 0, 255), thickness=2)
            thread_name, frame, results = result_queue.get(timeout=30)  # Wait for results with a timeout
            if frame is None:  # Check for completion signal
                break

            # Process the detection results (e.g., draw boxes, save frame, etc.)
            print(f"Processing results from {thread_name}")
            for point_center in results:
                if thread_name == "Thread_1":
                    cv2.circle(image_bgr, point_center, 50, (165, 0, 255), -1)
                else:
                    cv2.circle(image_bgr, point_center, 50, (0, 165, 255), -1)
            
            cv2.imwrite("check.jpg", image_bgr)
            cv2.imwrite(thread_name + ".jpg", frame)



        except Queue.Empty:
            continue

if __name__ == "__main__":
    # Create a queue to communicate results between threads
    result_queue = Queue()

    # Define the video paths
    video_paths = ["/home/sonlt373/Desktop/SoNg/Multi_cam_tracking/dev/multicam_tracking/sample_data/x5sonnt.mp4", 
                   "/home/sonlt373/Desktop/SoNg/Multi_cam_tracking/dev/multicam_tracking/sample_data/x6sonnt.mp4"
                   ]

    # Create and start the threads
    threads = []
    for i, video_path in enumerate(video_paths):

        if i == 0:
            homo_matrix = np.loadtxt('./sample_data/frame_video_1/matrix_homo.txt')
        elif i == 1:
            homo_matrix = np.loadtxt('./sample_data/frame_video_2/matrix_homo.txt')

        thread = threading.Thread(target=process_video, args=(video_path, result_queue, f"Thread_{i+1}", homo_matrix))
        thread.start()
        threads.append(thread)

    # Process results in the main thread
    process_results(result_queue)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()
