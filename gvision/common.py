import json
import cv2

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data

def warm_up_model(Pkg_Detector):
    img1 = cv2.imread("./data_warmup/frame_0.jpg")
    img2 = cv2.imread("./data_warmup/frame_1.jpg")
    img3 = cv2.imread("./data_warmup/frame_2.jpg")
    img4 = cv2.imread("./data_warmup/frame_3.jpg")
    img5 = cv2.imread("./data_warmup/frame_4.jpg")
    img6 = cv2.imread("./data_warmup/frame_5.jpg")
    img7 = cv2.imread("./data_warmup/frame_6.jpg")

    list_img = [img1, img2, img3, img4, img5, img6, img7]
    Pkg_Detector.detect_bbox(list_img)
