import cv2
import numpy as np
from queue import Queue
import json
import base64
import time
from engine import queues

def visualize_result(result_queue, IMAGE_OVERVIEW):
    while True:
        try:
            frame_list, result_tracking, dict_bboxes_homo_all_frame = result_queue.get(timeout=30)
            final_data = []

            IMAGE_OVERVIEW_ = IMAGE_OVERVIEW.copy()
            for bboxes in result_tracking:
                IMAGE_OVERVIEW_ = cv2.putText(IMAGE_OVERVIEW_, f'Pkg:{str(bboxes[4])}', (int(bboxes[0]), int(bboxes[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4 , cv2.LINE_AA)
                IMAGE_OVERVIEW_ = cv2.rectangle(IMAGE_OVERVIEW_, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])), (0, 0, 255), 3 )
            
            cv2.rectangle(IMAGE_OVERVIEW_, (0, 0), (IMAGE_OVERVIEW_.shape[1], IMAGE_OVERVIEW_.shape[0]), (255, 255, 255), 50)
            _, buffer = cv2.imencode('.jpg', IMAGE_OVERVIEW_)

            overview_data = base64.b64encode(buffer).decode('utf-8')

            for frame in frame_list:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                final_data.append(frame_data)
            data = json.dumps({'frames': final_data, 'overview': overview_data})
            yield f"data: {data}\n\n"

            while result_queue.qsize() > queues.SIZE_QUEUE_TRACKING // 2:
                try:
                    result_queue.get_nowait()
                except Empty:
                    break


        except Queue.Empty:
            continue
