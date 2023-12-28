import cv2
import logging
import numpy as np
from collections import deque
from typing import Deque

class MotionDetector:
    def __init__(self, buffer_size: int = 5, threshold: int = 25):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.frames: Deque[np.ndarray] = deque(maxlen=self.buffer_size)

    def update_frame(self, frame: np.ndarray):
        logging.info("Updating frame in motion detector...")
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frames.append(gray_frame)

    def is_motion_detected(self) -> bool:
        logging.info("Analysing frames for motion...")
        if len(self.frames) < self.buffer_size:
            return False

        avg_frame = np.mean(np.stack(self.frames, axis=0), axis=0).astype(np.uint8)
        frame_delta = cv2.absdiff(self.frames[-1], avg_frame)
        _, thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)
        return np.sum(thresh) > 0
