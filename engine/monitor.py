import time
from engine import queues

def monitor_queues(interval=10):
    while True:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("-"*20)
        print(f"Time: {current_time}")
        print(f"FPS EACH PROCESS VIDEO_CAPTURE X PKG_DETECT X POST_PROCESS: {[queues.FPS_EACH_PROCESS[key] for key in queues.FPS_EACH_PROCESS.keys()]}")
        print(f"QUEUE_CAPTURE_VIDEO size: {[q.qsize() for q in queues.QUEUE_CAPTURE_VIDEO]}")
        print(f"RESULT_QUEUE_DETECTION size: {queues.RESULT_QUEUE_DETECTION.qsize()}")
        print(f"RESULT_QUEUE_TRACKING size: {queues.RESULT_QUEUE_TRACKING.qsize()}")
        time.sleep(interval)
