import logging
from typing import Tuple
from picamera2 import Picamera2
from src.detection import CatDetector


class Camera(object):
    def __init__(self, detector: CatDetector, frame_size: Tuple[int, int] = (640, 480), buffer_size: int = 5):
        self.detector = detector
        self.picam2 = Picamera2()
        self.frame_size = frame_size
        camera_config = self.picam2.create_still_configuration(main={"size": self.frame_size})
        self.picam2.configure(camera_config)
        self.picam2.start()
    

    def capture_and_detect(self):
        logging.info("Capturing frame for object detection...")
        frame = self.picam2.capture_array()
        detection_type, detected_image, _ = self.detector.detect_cat_or_person(frame)
        return detection_type, detected_image, frame
