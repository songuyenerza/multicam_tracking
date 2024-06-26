from flask import render_template, Response
from app import app
from app.visualize import visualize_result
from engine.queues import RESULT_QUEUE_TRACKING
import cv2
from gvision.constants import *

IMAGE_OVERVIEW = cv2.imread("./data_wheel_daitu/overview_image.jpg")
IMAGE_OVERVIEW = cv2.resize(IMAGE_OVERVIEW, (WIDTH_IMAGE_OVERVIEW, HEIGHT_IMAGE_OVERVIEW))



@app.route('/view')
def index():
    print("Rendering index.html")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(visualize_result(RESULT_QUEUE_TRACKING, IMAGE_OVERVIEW), mimetype='text/event-stream')
