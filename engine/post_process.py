from queue import Queue
import numpy as np
import time
from gvision.algorithm.algorithm import Warp_box_to_overview, match_points_across_frames
from gvision.constants import *
from engine import queues

def post_process(Tracker, RESULT_QUEUE_DETECTION, RESULT_QUEUE_TRACKING, ROI_OVERVIEW):

    while True:
        try:
            t_start = time.time()
            frame_list, dict_bboxes_homo_all_frame = RESULT_QUEUE_DETECTION.get(timeout=30)
            if len(frame_list) == 0:
                break
            #   warp to over_view image and filter out roi
            dict_bboxes_all_frame = Warp_box_to_overview(dict_bboxes_homo_all_frame, ROI_OVERVIEW)

            #    matching box giữa các frame
            matched_points = match_points_across_frames(dict_bboxes_all_frame, threshold = 200)
            # print("matched_points: ", matched_points)

            result_box = []
            size_box = SIZE_BOX
            for match in matched_points:
                if len(match) == 2:
                    p1, p2 = match
                    box_xyxy = [int((p1[0] + p2[0]) * 0.5 - size_box / 2),
                                int((p1[1] + p2[1]) * 0.5 - size_box / 2),
                                int((p1[0] + p2[0]) * 0.5 + size_box / 2),
                                int((p1[1] + p2[1]) * 0.5 + size_box / 2)]
                else:
                    p1 = match[0]
                    box_xyxy = [int(p1[0] - size_box / 2),
                                int(p1[1] - size_box / 2),
                                int(p1[0] + size_box / 2),
                                int(p1[1] + size_box / 2)]
                result_box.append(box_xyxy)
            outputs_tracking = Tracker.update(np.array(result_box), np.array([1] * len(result_box)))
            result_tracking = [[output.tlwh[0], output.tlwh[1], output.tlwh[0] + output.tlwh[2], output.tlwh[1] + output.tlwh[3], output.track_id] for output in outputs_tracking]
            

            # if RESULT_QUEUE_TRACKING.qsize() == queues.SIZE_QUEUE_TRACKING:
            #     RESULT_QUEUE_TRACKING.get_nowait()
            
            RESULT_QUEUE_TRACKING.put((frame_list, result_tracking, dict_bboxes_homo_all_frame))

            queues.FPS_EACH_PROCESS["Post_process"] = 1 / (time.time() - t_start)

        except Queue.Empty:
            continue
