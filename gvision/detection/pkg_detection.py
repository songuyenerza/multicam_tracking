from ultralytics import YOLO
from pathlib import Path
import numpy as np
import concurrent.futures

class AttributeObjDetection:
    def __init__(self, model_dir: Path,
                image_size: int = 640,
                conf: float = 0.3,
                iou: float = 0.4,
                device: int = -1,
                ) -> None:
        """
        instanciate the model.

        Parameters
        ----------
        model_dir : Path
            directory where to find the model weights.

        device : str
            the device name to run the model on.
        """
        self.model = YOLO(model_dir)
        self.class_name = self.model.names

        self.image_size = image_size
        self.conf = conf
        self.iou = iou
        if device < 0:
            self.device = 'cpu'
        else:
            self.device = 'cuda:{0}'.format(device)

    def detect_bbox(self, img_list):
        # Make predictions
        predictions = self.model.predict(
            imgsz= self.image_size,
            source=img_list,
            conf= self.conf,
            iou = self.iou,
            device = self.device,
            verbose=False,
            stream = True
        )
        result_detection = []

        for pred_item in predictions:
            result_detection.append([[int(num) for num in box.xyxy[0]] for box in pred_item.to("cpu").numpy().boxes])

        return result_detection
