import os
import cv2
import math
from picamera2 import Picamera2
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv


picam2 = Picamera2()
camera_config = picam2.create_still_configuration(
    main={
        "size": (
            1920, 1080)}, lores={
                "size": (
                    640, 480)}, display="lores")
picam2.configure(camera_config)
picam2.start()

load_dotenv()
model_path = os.getenv('MODEL_PATH')

model = YOLO(model_path)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    img = picam2.capture_array()
    results = model.predict(source=img, save=False)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(
                x2), int(y2)  # convert to int values

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(
                img,
                classNames[cls],
                org,
                font,
                fontScale,
                color,
                thickness)

    cv2.imshow("Detection", img)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
