from gvision.constants import *
import cv2
import numpy as np
import time
from queue import Queue
from engine import queues
import os

# Function visualize result output 
def visualize_result(RESULT_QUEUE_TRACKING, IMAGE_OVERVIEW):
    print("[IMAGE_OVERVIEW] :: ", WIDTH_IMAGE_OVERVIEW, HEIGHT_IMAGE_OVERVIEW)

    grid_image = np.zeros((2160, 7680, 3), dtype=np.uint8)
    count_id = 0
    while True:
        try:
            t_start = time.time()

            IMAGE_OVERVIEW_ = IMAGE_OVERVIEW.copy()
            frame_list, result_tracking, dict_bboxes_homo_all_frame = RESULT_QUEUE_TRACKING.get(timeout=30) 
            # Determine the size of each frame in the 2x4 grid
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
            count_id += 1
            cv2.imwrite(os.path.join("./data_wheel_daitu/output_checker", f"over_view_{count_id}.jpg"), final_image)

            # Check the size of RESULT_QUEUE_TRACKING and clear if necessary
            while RESULT_QUEUE_TRACKING.qsize() > queues.SIZE_QUEUE_TRACKING // 2:
                try:
                    RESULT_QUEUE_TRACKING.get_nowait()
                except Empty:
                    break

        except TimeoutError:
            continue