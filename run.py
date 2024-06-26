from app import app
import threading
from engine.video_capture import VideoCapture
from engine.pkg_detection import pkg_detection
from engine.post_process import post_process
from engine.visualize import visualize_result
from engine.monitor import monitor_queues
from engine import queues

from gvision.algorithm.algorithm import Read_roi_file
from gvision.detection.pkg_detection import AttributeObjDetection
from gvision.common import read_json_file, warm_up_model
from gvision.constants import *
from queue import Queue
import numpy as np
import cv2

import sys
sys.path.append('./gvision')
from gvision.tracker.byte_tracker import BYTETracker

if __name__ == '__main__':
    source_list = read_json_file("source.json")
    threads = []

    ROI_OVERVIEW = Read_roi_file("./data_wheel_daitu/roi_overview_image.txt", WIDTH_IMAGE_OVERVIEW, HEIGHT_IMAGE_OVERVIEW)
    IMAGE_OVERVIEW = cv2.imread("./data_wheel_daitu/overview_image.jpg")
    IMAGE_OVERVIEW = cv2.resize(IMAGE_OVERVIEW, (WIDTH_IMAGE_OVERVIEW, HEIGHT_IMAGE_OVERVIEW))

    Tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30)
    Pkg_Detector = AttributeObjDetection(model_dir='./gvision/weights/240124_yolov8s_package_640.pt', image_size=640, conf=0.4, iou=0.4, device=0)
    warm_up_model(Pkg_Detector)

    for i, video_info in enumerate(source_list):
        result_queue = Queue(maxsize=queues.SIZE_QUEUE_CAPTURE)
        queues.QUEUE_CAPTURE_VIDEO.append(result_queue)
        video = video_info["video"]
        homo_matrix = np.loadtxt(video_info["homography_matrix"])
        thread = threading.Thread(target=VideoCapture, args=(video, result_queue, f"Stream_{i + 1}", homo_matrix))
        thread.start()
        threads.append(thread)
    
    thread = threading.Thread(target=pkg_detection, args=(Pkg_Detector, queues.QUEUE_CAPTURE_VIDEO, queues.RESULT_QUEUE_DETECTION))
    thread.start()
    threads.append(thread)
    
    thread = threading.Thread(target=post_process, args=(Tracker, queues.RESULT_QUEUE_DETECTION, queues.RESULT_QUEUE_TRACKING, ROI_OVERVIEW))
    thread.start()
    threads.append(thread)

    # Start the monitor thread
    monitor_thread = threading.Thread(target=monitor_queues)
    monitor_thread.start()
    threads.append(monitor_thread)

    # visualize_result save video
    thread = threading.Thread(target=visualize_result, args=(queues.RESULT_QUEUE_TRACKING, IMAGE_OVERVIEW, ))
    thread.start()
    threads.append(thread)

    ## visualize_result to web
    # app.run()

    for thread in threads:
        thread.join()
