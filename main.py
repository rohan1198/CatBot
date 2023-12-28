import os
import cv2
import logging
import asyncio
from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder
from src.camera import Camera
from src.detection import CatDetector
from src.telegram_notifier import TelegramNotifier


async def periodic_detection(bot, chat_id):
    while True:
        logging.info("Starting object detection cycle...")
        detection_count = {'cat': 0, 'person': 0, 'none': 0}

        for i in range(10):  # Perform 10 detections
            logging.info(f"Object detection iteration: [{i+1}/10]")
            detection_type, img_with_box, img_original = camera.capture_and_detect()  # Capture and detect returns original image as well
            detection_count[detection_type] += 1

            if detection_count['cat'] > 5 or detection_count['person'] > 5:
                logging.info(f"Majority detection confirmed: {detection_type}")
                success, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))  # Send original colored image
                if success:
                    await bot.send_photo(chat_id=chat_id, photo=encoded_image.tobytes(), caption=f"{detection_type.capitalize()} detected!")
                    break

        logging.info("Cycle complete. Sleeping for 30 seconds")
        await asyncio.sleep(30)  # Wait for 30 seconds before the next cycle

def start_bot_polling(application):
    asyncio.create_task(application.run_polling())

def main():
    # Load environment variables
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    default_chat_id = os.getenv('TELEGRAM_CHAT_ID')

    # Create the application
    application = ApplicationBuilder().token(bot_token).build()
    bot = application.bot

    # Initialize the asyncio event loop
    loop = asyncio.get_event_loop()

    # Schedule the periodic detection task and start the bot polling
    loop.create_task(periodic_detection(bot, default_chat_id))
    loop.run_until_complete(application.run_polling())

if __name__ == '__main__':
    # Enable logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    load_dotenv()

    # Initialize the CatDetector and Camera
    model_path = os.getenv('MODEL_PATH')
    detector = CatDetector(model_path)
    camera = Camera(detector)  # Initialize camera with grayscale processing

    main()
