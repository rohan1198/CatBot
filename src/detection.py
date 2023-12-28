import cv2
import numpy as np
from typing import Tuple, Optional
from ultralytics import YOLO


class CatDetector(object):
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.cat_class_id = 15
        self.person_class_id = 0


    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize the imahe to the target size for faster processing

        Args:
            image (np.ndarray): The original image
            target_size (Tuple[int, int]): The target width and height

        Returns:
            np.ndarray: The resized image
        """
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    

    def detect_cat_or_person(self, image: np.ndarray, resize_to: Optional[Tuple[int, int]] = None) -> Tuple[str, Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Detects a cat in the given image with optional resizing for efficiency

        Args:
            image (np.ndarray): Image array
            resize_to (Optional[Tuple[int, int]]): Resize image for faster processing

        Returns:
            A tuple containing a boolean indicating if a cat is detected, the image with
            the bounding box, and the bounding box coordinates.
        """
        try:
            if resize_to:
                image = self.resize_image(image, resize_to)

            results = self.model.predict(source=image, save=False)

            for r in results:
                for box in r.boxes:
                    detected_class = int(box.cls[0])
                    if detected_class == self.cat_class_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        return "cat", image, (x1, y1, x2, y2)
                    elif detected_class == self.person_class_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 3)
                        return "person", image, (x1, y1, x2, y2)

            return "none", None, None

        except Exception as e:
            print(f"Error while detecting the cat or person: {e}")
            return "error", None, None
