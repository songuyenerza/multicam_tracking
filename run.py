import threading
import cv2
from ultralytics import YOLO
from queue import Queue
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

import sys
sys.path.append('./gvision')
from gvision.tracker.byte_tracker import BYTETracker

H = 700
W  = 4408

with open('./sample_data/roi_fullview.txt', 'r') as file:
    roi_data = file.readline().strip().split(';')
    roi_data = [float(x) for x in roi_data]

# Convert ROI data into pairs
roi_pairs = [(roi_data[i] * W, roi_data[i + 1] * H) for i in range(0, len(roi_data), 2)]
roi_pairs = [(int(x), int(y)) for x, y in roi_pairs]
print("roi_pairs: ", roi_pairs)
# Initialize the YOLO model
model = YOLO('./pretrained/240124_yolov8s_package_640.pt')

# save video output
OUTPUT_VIDEO = "video_output_1.mp4"
FPS = 15

# init tracker
track_thresh  = 0.5
track_buffer = 30
match_thresh = 0.8
Tracker_bytetrack = BYTETracker(track_thresh, track_buffer , match_thresh, frame_rate=30)


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

# Define a function to process video frames
def process_video(video_path, result_queue, thread_name, homo_matrix):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS: ", fps)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if thread_name == "Thread_1" and frame_count < 0:
            frame_count += 1
            continue
        predictions = model.predict(source = frame,
                                imgsz = 640,
                                conf = 0.3,
                                iou = 0.2,
                                verbose = False,
                                save = False)
        print("---------------------------------------")    
        results = []
        
        for i, box in enumerate(predictions[0].boxes):
            cls = int(box.cls)
            box_xyxy = [float(num) for num in box.xyxy[0]]

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
            point_center_x = (transformed_top_left[0] + transformed_bottom_right[0]) / 2
            point_center_y = (transformed_top_left[1] + transformed_bottom_right[1]) / 2
            point_center = (point_center_x, point_center_y)

            if thread_name == "Thread_1":
                point_center = (int(point_center[0] - 10), int(point_center[1]))
                if W > point_center[0] > 2146 and 100 < point_center[1] < H:
                    if is_point_in_polygon(point_center, roi_pairs):
                        results.append(point_center)

            else:
                point_center = (int(point_center[0]), int(point_center[1]))
                if point_center[0] < 2148 and 100 < point_center[1] < H:
                    if is_point_in_polygon(point_center, roi_pairs):
                        results.append(point_center)

        # Put the results in the queue
        result_queue.put((thread_name, frame, results))

    cap.release()
    result_queue.put((thread_name, None, None))

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

def post_process(result_queues):
    # save video output

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_save = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (3840, 2298)) 

    blink_interval_1 = 0.5  # Blinking interval in seconds
    blink_state_1 = True  # Initial state of the blinking light
    last_blink_time_1 = time.time()

    blink_interval_2 = 0.5  # Blinking interval in seconds
    blink_state_2 = True  # Initial state of the blinking light
    last_blink_time_2 = time.time()

    while True:
        try:
            image_bgr = np.ones((H, W, 3), dtype=np.uint8) * 128
            cv2.line(image_bgr, (2146, 0), (2146, 300), (0, 0, 255), thickness=2)
            dict_points_all_frame = {}

            frame1, frame2 = None, None
            for result_queue in result_queues:
                thread_name, frame, results = result_queue.get(timeout=30)  # Wait for results with a timeout
                if frame is None:  # Check for completion signal
                    break
                
                # todo: sonnt
                if len(results)> 0:
                    dict_points_all_frame[thread_name] = results
                if thread_name == "Thread_1":
                    homo_matrix = np.loadtxt('./sample_data/frame_video_1/matrix_homo.txt')
                    warped_image1 = cv2.warpPerspective(frame, homo_matrix, (W, H))
                    frame1 = frame
                elif thread_name == "Thread_2":
                    homo_matrix = np.loadtxt('./sample_data/frame_video_2/matrix_homo.txt')
                    warped_image2 = cv2.warpPerspective(frame, homo_matrix, (W, H))
                    frame2 = frame

            overview_image = np.ones((H, W, 3), dtype=np.uint8)* 150

            overview_image_BGR = cv2.imread("./sample_data/full_view.jpg")
            overview_image_BGR = cv2.resize(overview_image_BGR, (W, H))

            overview_image[:, :2146] = warped_image2[:, :2146]
            overview_image[:, 2146:] = warped_image1[:, 2146:]
               
            # process dict_points_all_frame
            matched_points = match_points_across_frames(dict_points_all_frame)
            result_box = []
            size_box = 150
            for match in matched_points:
                if len(match) == 2:
                    p1, p2 = match
                    cv2.circle(overview_image, p1, 10, (255, 0, 0), -1)
                    cv2.circle(overview_image, p2, 10, (255, 0, 0), -1)
                    cv2.line(overview_image, p1, p2, (255, 255, 255), 10)
                    
                    box_xyxy = [int((p1[0] + p2[0]) * 0.5 - size_box/2),
                                  int((p1[1] + p2[1]) * 0.5 - size_box/2),
                                  int((p1[0] + p2[0]) * 0.5 + size_box/2),
                                  int((p1[1] + p2[1]) * 0.5 + size_box/2)]
                else:
                    p1 = match[0]
                    cv2.circle(overview_image, p1, 20, (0, 127, 255), -1)
                    box_xyxy = [int(p1[0] - size_box/2),
                                int(p1[1] - size_box/2),
                                int(p1[0] + size_box/2),
                                int(p1[1] + size_box/2)]
                result_box.append(box_xyxy)
            # tracking 
            outputs_tracking = Tracker_bytetrack.update(np.array(result_box), np.array([1] * len(result_box)))

            for output in outputs_tracking:
                bboxes = output.tlwh
                id = output.track_id
                bboxes = [bboxes[0], bboxes[1] , bboxes[0] + bboxes[2], bboxes[1] + bboxes[3], id]
                overview_image = cv2.putText(overview_image, f'Pkg:{str(id)}', (int(bboxes[0]), int(bboxes[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4 , cv2.LINE_AA)
                overview_image = cv2.rectangle(overview_image, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])), (0, 0, 255), 3 )

                overview_image_BGR = cv2.putText(overview_image_BGR, f'Pkg:{str(id)}', (int(bboxes[0]), int(bboxes[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4 , cv2.LINE_AA)
                overview_image_BGR = cv2.rectangle(overview_image_BGR, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])), (0, 0, 255), 3 )

                # blink led
                if 2800 < bboxes[0] < 3200:
                    current_time = time.time()
                    if current_time - last_blink_time_1 > blink_interval_1:
                        blink_state_1 = not blink_state_1
                        last_blink_time_1 = current_time
                    if blink_state_1:
                        cv2.circle(overview_image_BGR, (2900, 100), 40, (0, 255, 255), -1)

                if 700 < bboxes[0] < 1100:
                    current_time = time.time()
                    if current_time - last_blink_time_2 > blink_interval_2:
                        blink_state_2 = not blink_state_2
                        last_blink_time_2 = current_time
                    if blink_state_2:
                        cv2.circle(overview_image_BGR, (700, 100), 40, (0, 255, 255), -1)

            # draw to visualize
            # Draw bounding boxes around frame1 and frame2
            cv2.rectangle(frame1, (0, 0), (frame1.shape[1], frame1.shape[0]), (255, 255, 255), 30)
            cv2.rectangle(frame2, (0, 0), (frame2.shape[1], frame2.shape[0]), (255, 255, 255), 30)
            cv2.rectangle(overview_image, (0, 0), (overview_image.shape[1], overview_image.shape[0]), (255, 255, 255), 10)
            cv2.rectangle(overview_image_BGR, (0, 0), (overview_image_BGR.shape[1], overview_image_BGR.shape[0]), (255, 255, 255), 10)

            combined_top_frame = np.hstack((frame2, frame1))
            combined_width = combined_top_frame.shape[1]
            overview_image_BGR = cv2.resize(overview_image_BGR, (combined_width, int(overview_image.shape[0] * (combined_width / overview_image.shape[1]))))
            overview_image = cv2.resize(overview_image, (combined_width, int(overview_image.shape[0] * (combined_width / overview_image.shape[1]))))

            final_view = np.vstack((combined_top_frame, overview_image))
            final_view = np.vstack((final_view, overview_image_BGR))
            # cv2.imwrite("output.jpg", final_view)
            video_save.write(final_view)
            #   /////////////////
        except Queue.Empty:
            continue
    out.release()
    print("====== SUCCESS =====")

if __name__ == "__main__":

    result_queues = []
    video_paths = ["./sample_data/x5.mp4", 
                   "./sample_data/x6.mp4"
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

    post_process(result_queues)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
