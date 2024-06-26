from queue import Queue
import time
from engine import queues

def pkg_detection(pkg_detector, queue_capture_video, result_queue_pkgdetection):
    while True:
        try:
            t_start = time.time()
            images = []
            thread_name_list = []
            homo_matrix_list = []
            for result_queue in queue_capture_video:
                thread_name, frame, homo_matrix = result_queue.get(timeout=30)
                if frame is None:
                    break
                homo_matrix_list.append(homo_matrix)
                thread_name_list.append(thread_name)
                images.append(frame)

            pkg_bboxes = pkg_detector.detect_bbox(images)
            dict_bboxes_all_frame = {}
            for i, pkg_bbox in enumerate(pkg_bboxes):
                dict_bbox = {"bbox": pkg_bbox, "homo_matrix": homo_matrix_list[i]}
                dict_bboxes_all_frame[thread_name_list[i]] = dict_bbox

            # if result_queue_pkgdetection.qsize() == queues.SIZE_QUEUE_DETECTION:
            #     result_queue_pkgdetection.get()

            result_queue_pkgdetection.put((images, dict_bboxes_all_frame))

            queues.FPS_EACH_PROCESS["Pkg_detection"] = 1 / (time.time() - t_start)
            # print("=====> FPS Pkg_Detection : " ,  1 / (time.time() - t_start))
        except Queue.Empty:
            continue
