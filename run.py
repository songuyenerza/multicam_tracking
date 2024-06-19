import threading
import cv2
from ultralytics import YOLO
from queue import Queue
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

# Initialize the YOLO model
model = YOLO('./pretrained/240124_yolov8s_package_640.pt')
H = 300
W  = 6400
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
                                conf = 0.5,
                                iou = 0.6,
                                verbose = False,
                                save = False)
        print("---------------------------------------")    
        results = []
        
        for i, box in enumerate(predictions[0].boxes):
            cls = int(box.cls)
            box_xyxy = [float(num) for num in box.xyxy[0]]

            transformed_box = np.dot(homo_matrix, np.array([box_xyxy[0], box_xyxy[1], 1]).reshape((3, 1)))
            point_center = transformed_box.reshape(1, 3)[0][:2]

            point_center = (int(point_center[0]), int(point_center[1]))
            if -1 < point_center[0] < W +1  and -1 < point_center[1] <  H + 1:
                if thread_name == "Thread_1":
                    if point_center[0] > 3700:
                        results.append(point_center)
                else:
                    results.append(point_center)


        # Put the results in the queue
        result_queue.put((thread_name, frame, results))

    cap.release()
    result_queue.put((thread_name, None, None))

def match_points_across_frames(dict_points_all_frame, threshold=200):
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
    print("matched_points: ", matched_points)
    return matched_points

def post_process(result_queues):
    while True:
        try:
            image_bgr = np.ones((H, W, 3), dtype=np.uint8) * 128
            cv2.line(image_bgr, (3200, 0), (3200, 300), (0, 0, 255), thickness=2)
            dict_points_all_frame = {}
            for result_queue in result_queues:
                thread_name, frame, results = result_queue.get(timeout=30)  # Wait for results with a timeout
                if frame is None:  # Check for completion signal
                    break
                
                # todo: sonnt
                if len(results)> 0:
                    dict_points_all_frame[thread_name] = results
                
                cv2.imwrite(thread_name + ".jpg", frame)
               
            # process dict_points_all_frame
            print("dict_points_all_frame: ", dict_points_all_frame)
            matched_points = match_points_across_frames(dict_points_all_frame)

            for match in matched_points:
                if len(match) == 2:
                    p1, p2 = match
                    cv2.circle(image_bgr, p1, 30, (255, 0, 0), -1)
                    cv2.circle(image_bgr, p2, 30, (255, 0, 0), -1)

                    cv2.line(image_bgr, p1, p2, (255, 0, 0), 30)
                else:
                    p1 = match[0]
                    cv2.circle(image_bgr, p1, 50, (0, 255, 0), -1)


            cv2.imwrite("output.jpg", image_bgr)
        except Queue.Empty:
            continue

if __name__ == "__main__":
    result_queues = []

    video_paths = ["./sample_data/x5sonnt.mp4", 
                   "./sample_data/x6sonnt.mp4"
                   ]

    # Create and start the threads
    threads = []
    for i, video_path in enumerate(video_paths):
        result_queue = Queue()
        result_queues.append(result_queue)
        if i == 0:
            homo_matrix = np.loadtxt('./sample_data/frame_video_1/matrix_homo.txt')
        elif i == 1:
            homo_matrix = np.loadtxt('./sample_data/frame_video_2/matrix_homo.txt')

        thread = threading.Thread(target=process_video, args=(video_path, result_queue, f"Thread_{i+1}", homo_matrix))
        thread.start()
        threads.append(thread)

    # Process results in the main thread
    post_process(result_queues)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
