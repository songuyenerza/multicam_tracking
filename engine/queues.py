from queue import Queue

SIZE_QUEUE_TRACKING = 200
SIZE_QUEUE_DETECTION = 200
SIZE_QUEUE_CAPTURE = 200


RESULT_QUEUE_TRACKING = Queue(maxsize=50)
RESULT_QUEUE_DETECTION = Queue(maxsize=200)
QUEUE_CAPTURE_VIDEO = []

FPS_EACH_PROCESS = {"Video_capture": 0,
                    "Pkg_detection": 0,
                    "Post_process": 0}