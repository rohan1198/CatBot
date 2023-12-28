import cv2
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.camera import Camera
from src.detection import CatDetector
from utils.motion_detection import MotionDetector


def test_camera_initialization():
    detector_mock = Mock(spec=CatDetector)

    with patch('src.camera.Picamera2') as picam_mock:
        camera = Camera(detector_mock)

        assert camera.detector == detector_mock
        assert isinstance(camera.motion_detector, MotionDetector)

        picam_mock.assert_called_once()


def test_update_motion_detector():
    detector_mock = Mock(spec=CatDetector)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Dummy frame for picam
    resized_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Resized frame for cv2.resize, now 3-channel

    with patch('src.camera.Picamera2') as picam_mock, \
         patch('cv2.resize', return_value=resized_frame) as resize_mock:
        picam_mock.return_value.capture_array.return_value = dummy_frame

        camera = Camera(detector_mock)
        returned_frame = camera.update_motion_detector()

        picam_mock.return_value.capture_array.assert_called_once()
        resize_mock.assert_called_with(dummy_frame, camera.frame_size, interpolation=cv2.INTER_AREA)
        assert np.array_equal(returned_frame, resized_frame)  # Check if the returned frame is as expected


def test_is_motion_detected():
    detector_mock = Mock(spec=CatDetector)
    motion_detector_mock = Mock(spec=MotionDetector)
    with patch('src.camera.Picamera2'), \
         patch('src.camera.MotionDetector', return_value=motion_detector_mock):

        camera = Camera(detector_mock)
        camera.is_motion_detected()

        motion_detector_mock.is_motion_detected.assert_called_once()


def test_capture_and_detect():
    detector_mock = Mock(spec=CatDetector)
    with patch('src.camera.Picamera2') as picam_mock, \
         patch.object(detector_mock, 'detect_cat_or_person', return_value=('none', None, None)):

        camera = Camera(detector_mock)
        result = camera.capture_and_detect()

        picam_mock.return_value.capture_array.assert_called_once()
        detector_mock.detect_cat_or_person.assert_called_once()
        assert result == ('none', None, None)
