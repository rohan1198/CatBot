import cv2
import os
from dotenv import load_dotenv
from src.camera import Camera
from src.detection import CatDetector
from src.telegram_notifier import TelegramNotifier


def main():
    # Load environment variables
    load_dotenv()
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    model_path = "/home/raspi/CatBot/models/yolov8n.pt"

    # Initialize the CatDetector and Camera
    detector = CatDetector(model_path)
    camera = Camera(detector)

    # Initialize Telegram Notifier
    notifier = TelegramNotifier(bot_token, chat_id)

    # Capture image
    print("Capturing image...")
    _, image, _ = camera.capture_and_detect()

    # Send test message
    print("Sending test message...")
    notifier.send_message("Hello, this is a test message from Raspberry Pi!")

    # Send captured image
    if image is not None:
        print("Sending captured image...")
        notifier.send_photo(
            image, "This is a test image captured by the Raspberry Pi camera.")
    else:
        print("No image captured.")


if __name__ == '__main__':
    main()
