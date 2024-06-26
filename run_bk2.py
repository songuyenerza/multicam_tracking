from gvision.algorithm.algorithm import is_point_in_polygon, match_points_across_frames, Warp_box_to_overview
from gvision.detection.pkg_detection import AttributeObjDetection
from gvision.common import read_json_file, warm_up_model
from gvision.constants import *

import threading
import cv2
from queue import Queue
import time
import numpy as np
import random

import sys
sys.path.append('./gvision')
from gvision.tracker.byte_tracker import BYTETracker


# Define a function to capture video frames
def VideoCapture(video_path, result_queue, thread_name, homo_matrix):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS of video: {video_path} :: ", fps)

    while cap.isOpened():
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        result_queue.put((thread_name, frame, homo_matrix))
        if random.uniform(0, 1) < 0.1:
            print("=====> FPS VideoCapture : " ,  1 / (time.time() - t_start))

    cap.release()
    result_queue.put((thread_name, None))

# Function detect package on batch all frame
def pkg_detection(pkg_detector, queue_capture_video, result_queue_pkgdetection):
    while True:
        try:
            t_start = time.time()
            images = []
            thread_name_list = []
            homo_matrix_list = []
            for result_queue in queue_capture_video:
                thread_name, frame, homo_matrix = result_queue.get(timeout=30)  # Wait for results with a timeout
                if frame is None:  # Check for completion signal
                    break
                homo_matrix_list.append(homo_matrix)
                thread_name_list.append(thread_name)
                images.append(frame)

            pkg_bboxes = pkg_detector.detect_bbox(images)

            dict_bboxes_all_frame = {}
            for i, pkg_bbox in enumerate(pkg_bboxes):
                dict_bbox = {}
                dict_bbox["bbox"] = pkg_bbox
                dict_bbox["homo_matrix"] = homo_matrix_list[i]

                dict_bboxes_all_frame[thread_name_list[i]] = dict_bbox
                
            result_queue_pkgdetection.put((images, dict_bboxes_all_frame))
            time_postprocess = time.time() - t_start
            print("=====> FPS Pkg_Detection : " ,  1/ time_postprocess)
            #   /////////////////
        except Queue.Empty:
            continue

# Function implement logic matching box and tracking
def post_process(Tracker, RESULT_QUEUE_DETECTION, RESULT_QUEUE_TRACKING):

    while True:
        try:
            t_start = time.time()

            frame_list, dict_bboxes_homo_all_frame = RESULT_QUEUE_DETECTION.get(timeout=30)  # Wait for results with a timeout
            if len(frame_list) == 0:  # Check for completion signal
                break

            dict_bboxes_all_frame = Warp_box_to_overview(dict_bboxes_homo_all_frame)
            
            # #  process dict_points_all_frame
            matched_points = match_points_across_frames(dict_bboxes_all_frame)

            result_box = []
            size_box = SIZE_BOX
            for match in matched_points:
                if len(match) == 2:
                    p1, p2 = match
                    box_xyxy = [int((p1[0] + p2[0]) * 0.5 - size_box/2),
                                  int((p1[1] + p2[1]) * 0.5 - size_box/2),
                                  int((p1[0] + p2[0]) * 0.5 + size_box/2),
                                  int((p1[1] + p2[1]) * 0.5 + size_box/2)]
                else:
                    p1 = match[0]
                    box_xyxy = [int(p1[0] - size_box/2),
                                int(p1[1] - size_box/2),
                                int(p1[0] + size_box/2),
                                int(p1[1] + size_box/2)]
                result_box.append(box_xyxy)

            # tracking 
            outputs_tracking = Tracker.update(np.array(result_box), np.array([1] * len(result_box)))

            result_tracking = []
            for output in outputs_tracking:
                bboxes = output.tlwh
                bboxes = [bboxes[0], bboxes[1] , bboxes[0] + bboxes[2], bboxes[1] + bboxes[3], output.track_id]
                result_tracking.append(bboxes)
            
            # print("result_tracking: ", result_tracking)
            RESULT_QUEUE_TRACKING.put((frame_list, result_tracking, dict_bboxes_homo_all_frame))

            time_postprocess = time.time() - t_start
            print("=====> FPS Post_process: " ,  1/ time_postprocess)
            #   /////////////////
        except Queue.Empty:
            continue

# Function visualize result output 
def visualize_result(RESULT_QUEUE_TRACKING, IMAGE_OVERVIEW):
    print("[IMAGE_OVERVIEW] :: ", WIDTH_IMAGE_OVERVIEW, HEIGHT_IMAGE_OVERVIEW)

    grid_image = np.zeros((2160, 7680, 3), dtype=np.uint8)
    while True:
        try:
            t_start = time.time()

            IMAGE_OVERVIEW_ = IMAGE_OVERVIEW.copy()
            frame_list, result_tracking, dict_bboxes_homo_all_frame = RESULT_QUEUE_TRACKING.get(timeout=30) 


            # Determine the size of each frame in the 3x3 grid
            frame_height, frame_width = frame_list[0].shape[:2]
            grid_height = frame_height * 2
            grid_width = frame_width * 4
            # print("grid_height: ", grid_height, " grid_width: ", grid_width)
            # grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

            for i, frame in enumerate(frame_list):
                row = i // 4
                col = i % 4
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), 20)
                grid_image[row * frame_height: (row + 1) * frame_height, col * frame_width: (col + 1) * frame_width] = frame

            cv2.rectangle(grid_image, (0, 0), (grid_image.shape[1], grid_image.shape[0]), (255, 255, 255), 50)

            for bboxes in result_tracking:
                IMAGE_OVERVIEW_ = cv2.putText(IMAGE_OVERVIEW_, f'Pkg:{str(bboxes[4])}', (int(bboxes[0]), int(bboxes[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4 , cv2.LINE_AA)
                IMAGE_OVERVIEW_ = cv2.rectangle(IMAGE_OVERVIEW_, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])), (0, 0, 255), 3 )
            
            # Resize IMAGE_OVERVIEW_ to have the same width as the grid_image
            IMAGE_OVERVIEW_ = cv2.resize(IMAGE_OVERVIEW_, (grid_width, int(HEIGHT_IMAGE_OVERVIEW * (grid_width / WIDTH_IMAGE_OVERVIEW))))

            # Create a larger image to hold the grid and resized IMAGE_OVERVIEW_
            final_image = np.vstack((grid_image, IMAGE_OVERVIEW_))
            # cv2.imwrite("IMAGE_OVERVIEW.jpg", final_image)

            print("=====> FPS Visualize_result: " ,  1/ (time.time() - t_start))
        except Queue.Empty:
            continue

if __name__ == "__main__":
    # Queues include data after process videos (frame, bbox)
    QUEUE_CAPTURE_VIDEO = []
    RESULT_QUEUE_TRACKING = Queue()
    RESULT_QUEUE_DETECTION = Queue()

    max_queue_size = 100
    # init tracker
    track_thresh  = 0.5
    track_buffer = 30
    match_thresh = 0.8
    Tracker = BYTETracker(track_thresh, track_buffer , match_thresh, frame_rate=30)

    # Init model 
    Pkg_Detector = AttributeObjDetection(model_dir = './gvision/weights/240124_yolov8s_package_640.pt',
                                        image_size = 640,
                                        conf = 0.4,
                                        iou = 0.4,
                                        device = 0)
    # Warp up
    warm_up_model(Pkg_Detector)

    IMAGE_OVERVIEW = cv2.imread("./data_wheel_daitu/overview_image.jpg")
    IMAGE_OVERVIEW = cv2.resize(IMAGE_OVERVIEW, (WIDTH_IMAGE_OVERVIEW, HEIGHT_IMAGE_OVERVIEW))

    
    print("================= Init model Scucess ==============================")
    source_list  = read_json_file("source.json")

    print("source_list: ", source_list)

    # Create and start the threads
    threads = []
    for i, video_info in enumerate(source_list):
        result_queue = Queue(maxsize=max_queue_size)
        QUEUE_CAPTURE_VIDEO.append(result_queue)

        video = video_info["video"]
        homography_matrix_path = video_info["homography_matrix"]
        homo_matrix = np.loadtxt(homography_matrix_path)

        thread = threading.Thread(target=VideoCapture, args=(video, result_queue, f"Stream_{i+1}", homo_matrix))
        thread.start()
        threads.append(thread)

    # Thread Detect pkg
    thread = threading.Thread(target=pkg_detection, args=(Pkg_Detector, QUEUE_CAPTURE_VIDEO, RESULT_QUEUE_DETECTION))
    thread.start()
    threads.append(thread)

    # Thread post process:: Logic track on multi frame
    thread = threading.Thread(target=post_process, args=(Tracker, RESULT_QUEUE_DETECTION, RESULT_QUEUE_TRACKING))
    thread.start()
    threads.append(thread)

    # visualize result
    thread = threading.Thread(target=visualize_result, args=(RESULT_QUEUE_TRACKING, IMAGE_OVERVIEW, ))
    thread.start()
    threads.append(thread)
    
    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print("====== SUCCESS =====")