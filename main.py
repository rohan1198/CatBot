import os
import cv2
import time
import logging
import asyncio
import datetime
from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder
from src.camera import Camera
from src.detection import CatDetector
from src.telegram_notifier import TelegramNotifier

last_detection_time = 0

async def periodic_detection(bot, chat_id: str):
    """
    Perform periodic detection of cats and notify via Telegram

    Args:
        bot: Telegram bot instance for sending messages.
        chat_id (str): The chat ID for sending notifications.
    """
    global last_detection_time
    cooldown_period = 7200

    while True:
        try:
            current_time = time.time()
            current_hour = datetime.datetime.now().hour

            if 18 <= current_hour < 24:
                logging.info("Scheduled inactivity. Sleeping for 12 hours.")
                await asyncio.sleep(43200)
                continue

            if current_time - last_detection_time < cooldown_period:
                await asyncio.sleep(30)
                continue

            logging.info("Starting object detection cycle...")
            detection_count = {'cat': 0, 'person': 0, 'none': 0, 'error': 0}

            for i in range(10):
                logging.info(f"Object detection iteration: [{i+1}/10]")
                detection_type, img_with_box, img_original = camera.capture_and_detect()
                detection_count[detection_type] += 1

                if detection_type == "cat":
                    cv2.imwrite(f"data/cat_detected_{current_time}_{i}.jpg", img_original)

                # Send notification if majority detection is confirmed
                if detection_count['cat'] > 5 or detection_count['person'] > 5:
                    logging.info(f"Majority detection confirmed: {detection_type}")
                    success, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
                    if success:
                        await bot.send_photo(chat_id=chat_id, photo=encoded_image.tobytes(), caption=f"{detection_type.capitalize()} detected!")
                        break

                    last_detection_time = current_time

                    break

            logging.info("Cycle complete. Sleeping for 30 seconds")
            await asyncio.sleep(30)
        except Exception as e:
            logging.error(f"Error in periodic_detection: {e}")
            await bot.send_message(chat_id=chat_id, text=f"Error detected: {e}")


def start_bot_polling(application):
    """
    Start the bot polling.

    Args:
        application: The Telegram bot application instance.
    """
    asyncio.create_task(application.run_polling())


def main():
    """
    Main function to initialize and start the application.
    """
    load_dotenv()
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    default_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    model_path = os.getenv('MODEL_PATH')

    try:
        application = ApplicationBuilder().token(bot_token).build()
        bot = application.bot

        detector = CatDetector(model_path)
        global camera
        camera = Camera(detector)

        loop = asyncio.get_event_loop()
        loop.create_task(periodic_detection(bot, default_chat_id))
        loop.run_until_complete(application.run_polling())
    except Exception as e:
        logging.error(f"Error in main: {e}")


if __name__ == '__main__':
    main()
