import cv2
import logging
import numpy as np
from typing import Tuple, Optional
from ultralytics import YOLO

class CatDetector:
    """
    Class for detecting cats and persons in images using YOLO model.

    Attributes:
        model_path (str): Path to the YOLO model file.
        cat_class_id (int): Class ID for cats in the model.
        person_class_id (int): Class ID for persons in the model.
    """
    def __init__(self, model_path: str):
        try:
            self.model = YOLO(model_path)
            self.cat_class_id = 15
            self.person_class_id = 0
        except Exception as e:
            logging.error(f"CatDetector initialization failed: {e}")
            raise


    def detection(self, image: np.ndarray, resize_to: Optional[Tuple[int, int]] = None):
        """
        Detects a cat or person in the given image.

        Args:
            image (np.ndarray): The image to process.
            resize_to (Optional[Tuple[int, int]]): Resize the image for faster processing.

        Returns:
            tuple: Detection type, image with bounding box, and bounding box coordinates.
        """
        try:
            if resize_to:
                image = self.resize_image(image, resize_to)

            results = self.model.predict(source=image, save=False)
            for r in results:
                for box in r.boxes:
                    detected_class = int(box.cls[0])
                    if detected_class == self.cat_class_id:
                        return self._process_detection(box, "cat", image)
                    elif detected_class == self.person_class_id:
                        return self._process_detection(box, "person", image)
            return "none", None, None
        except Exception as e:
            logging.error(f"Object Detection Error: {e}")
            raise RuntimeError("ObjectDetectionError")


    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resizes the image to a specified target size.

        Args:
            image (np.ndarray): The original image.
            target_size (Tuple[int, int]): The target width and height.

        Returns:
            np.ndarray: The resized image.
        """
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


    @staticmethod
    def _process_detection(box, detection_type, image):
        """
        Process the detection and draw bounding box on the image.

        Args:
            box: Detected bounding box.
            detection_type (str): Type of detection ('cat' or 'person').
            image (np.ndarray): Image on which to draw the bounding box.

        Returns:
            tuple: Detection type, image with bounding box, and bounding box coordinates.
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (255, 0, 255) if detection_type == "cat" else (255, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        return detection_type, image, (x1, y1, x2, y2)
