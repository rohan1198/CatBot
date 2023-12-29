import cv2
import logging
import numpy as np
from picamera2 import Picamera2
from typing import Tuple
from src.detection import CatDetector


class Camera:
    """
    Camera class to handle image captures and passing images for detection.

    Attributes:
        detector (CatDetector): The object detection pipeline.
        picam2 (Picamera2): PiCamera instance for capturing images.
        frame_size (Tuple[int, int]): The resolution of the camera capture.
    """
    def __init__(self, detector: CatDetector, frame_size: Tuple[int, int] = (640, 640)):
        self.detector = detector
        self.picam2 = Picamera2()
        self.frame_size = frame_size

        try:
            camera_config = self.picam2.create_still_configuration(main={"size": self.frame_size})
            self.picam2.configure(camera_config)
            self.picam2.start()
        except Exception as e:
            logging.error(f"Camera initialization failed: {e}")
            raise

    def capture_and_detect(self):
        """
        Captures an image from the camera and processes it for object detection.

        Returns:
            tuple: The type of detection, processed image with bounding box, and original image.
        """
        logging.info("Capturing frame for object detection...")
        try:
            frame = self.picam2.capture_array()

            # Ensure frame is in the correct format (8-bit per channel)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            detection_type, detected_image, bbox = self.detector.detect_cat_or_person(frame)
            return detection_type, detected_image, frame
        except Exception as e:
            logging.error(f"Error in capture_and_detect: {e}")
            return "error", None, None
