import cv2
import numpy as np
from typing import Tuple, Optional

class MotionDetector(object):
    def __init__(self, threshold: float = 25.0, min_area: int = 500):
        """
        Initialize the MotionDetector

        Args:
            threshold (float): The threshold for detecting motion.
            min_area (int): The minimum area of the contours to consider as motion.
        """
        self.threshold = threshold
        self.min_area = min_area
        self.previous_frame = None

    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        DEetect motion in the given frame.

        Args:
            frame (np.ndarray): The frame in which to detect motion.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: A tuple containing a boolean indicating if
                                               motion is detected and the threshold image.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_frame is None:
            self.previous_frame = gray
            return False, None
        
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue

            self.previous_frame = gray
            return True, thresh
        
        self.previous_frame = gray
        return False, None
