import cv2
import time
import random
from engine import queues
from gvision.constants import *

def VideoCapture(video_path, result_queue, thread_name, homo_matrix):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS of video: {video_path} :: ", fps)
    frame_count = 0
    frame_skip = 0

    # Adjust initial frame skips based on thread_name
    if thread_name == "Stream_1":
        frame_skip = 8
    elif thread_name == "Stream_4":
        frame_skip = 13
    elif thread_name == "Stream_7":
        frame_skip = 9
    elif thread_name == "Stream_6":
        frame_skip = 3

    elif thread_name == "Stream_3":
        frame_skip = 4

    while cap.isOpened():
        t_start = time.time()
        
        # Skip initial frames if needed
        if frame_count < frame_skip:
            frame_count += 1
            cap.grab()  # Skip this frame
            continue

        if thread_name == "Stream_6":
            pass 
        else:
            if  frame_count % 2 == 0:
                frame_count += 1
                ret, frame = cap.read()
                continue

        ret, frame = cap.read()

        if not ret:
            break

        # if result_queue.qsize() == queues.SIZE_QUEUE_CAPTURE:
        #     result_queue.get_nowait()
            
        result_queue.put((thread_name, frame, homo_matrix))

        queues.FPS_EACH_PROCESS["Video_capture"] = 1 / (time.time() - t_start)
        frame_count += 1

    cap.release()
    result_queue.put((thread_name, None, None))
