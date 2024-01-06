import cv2
import logging
import numpy as np
from picamera2 import Picamera2
from typing import Tuple
from src.detection import CatDetector
from utils.motion_detection import MotionDetector


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
        self.motion_detector = MotionDetector()

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
        logging.info("Capturing frame for motion detection...")
        try:
            frame = self.picam2.capture_array()

            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            motion_detected, _ = self.motion_detector.detect_motion(frame)
            if motion_detected:
                logging.info("Motion detected. Processing frame for object detection...")
                detection_type, detected_image, _ = self.detector.detection(frame)
                return detection_type, detected_image, frame
            else:
                return "no motion", None, None
        except Exception as e:
            logging.error(f"Camera or Motion Detection Error: {e}")
            raise RuntimeError("CameraError")
