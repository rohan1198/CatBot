import logging
import numpy as np
from typing import Tuple, Optional
from picamera2 import Picamera2
from src.detection import CatDetector


class Camera(object):
    """
    Camera class for capturing images and detecting objects using the CatDetector class

    Attributes:
        detector (CatDetector): An instance of CatDetector for object detection.
        picam2 (Picamera2): Instance of PiCamera2 for capturing images
        frame_size (Tuple[int, int]): The resolution of the camera.
    """
    
    def __init__(self, detector: CatDetector, frame_size: Tuple[int, int] = (640, 480), buffer_size: int = 5):
        """
        Initialize the Camera object with a CatDetector and frame size.

        Args:
            detector (CatDetector): An instance of CatDetector
            frame_size (Tuple[int, int]): Frame resolution, defaults to 640x480
        """
        self.detector = detector
        self.picam2 = Picamera2()
        self.frame_size = frame_size
        camera_config = self.picam2.create_still_configuration(main={"size": self.frame_size})
        self.picam2.configure(camera_config)
        self.picam2.start()
    

    def capture_and_detect(self) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture an image from the camera and detect objects.

        Returns:
            Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing the detection type, 
            the image with detection box, and the original image.
        """
        logging.info("Capturing frame for object detection...")
        frame = self.picam2.capture_array()
        detection_type, detected_image, _ = self.detector.detect_cat_or_person(frame)

        return detection_type, detected_image, frame
