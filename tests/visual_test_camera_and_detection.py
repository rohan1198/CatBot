import os
import cv2
import time
from dotenv import load_dotenv
from src.camera import Camera
from src.detection import CatDetector


def main():
    model_path = os.getenv('MODEL_PATH')

    detector = CatDetector(model_path)
    camera = Camera(detector)

    try:
        while True:
            camera.update_motion_detector()
            if camera.is_motion_detected():
                print("Motion detected! Checking for cats/persons...")
                detection_type, detected_image, _ = camera.capture_and_detect()

                if detection_type in ["cat", "person"]:
                    print(f"{detection_type.capitalize()} detected!")
                    cv2.imshow("Detection", detected_image)
                else:
                    print("No cat or person detected.")

            else:
                print("No motion detected.")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(1)  # Wait for a bit before next frame

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    load_dotenv()
    main()
