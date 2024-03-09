import os
import cv2
import time
import json
import logging
import asyncio
import datetime
from pathlib import Path
from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder
from telegram.ext import CommandHandler
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

            if current_time - last_detection_time < cooldown_period:
                await asyncio.sleep(30)
                continue

            logging.info("Starting object detection cycle...")
            detection_count = {'cat': 0, 'person': 0, 'none': 0, 'error': 0}

            for i in range(5):
                logging.info(f"Object detection iteration: [{i+1}/5]")
                detection_type, img_with_box, img_original = camera.capture_and_detect()
                detection_count[detection_type] += 1

                if detection_type == "cat":
                    file_name = f"cat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
                    file_path = Path(f"/home/raspi/CatBot/data/detections/{datetime.datetime.now().strftime('%Y%m%d')}/{file_name}")
                    try:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(file_path), img_original):
                            raise IOError("Failed to write image to file")
                    except Exception as e:
                        logging.error(f"Error saving cat image: {e}")

                    metadata = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "detection_type": detection_type,
                        "file_name": str(file_path)
                    }

                    with open(str(file_path.with_suffix(".json")), "w") as f:
                        json.dump(metadata, f)

                # Send notification if majority detection is confirmed
                if detection_count['cat'] > 3 or detection_count['person'] > 3:
                    logging.info(f"Majority detection confirmed: {detection_type}")
                    success, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB))
                    if success:
                        await bot.send_photo(chat_id=chat_id, photo=encoded_image.tobytes(), caption=f"{detection_type.capitalize()} detected!")
                        break

                    logging.info("Sleeping for 1 hour after detection.")
                    await asyncio.sleep(3600)

                    last_detection_time = current_time
                    break

            logging.info("Cycle complete. Sleeping for 30 seconds")
            await asyncio.sleep(30)
        
        except RuntimeError as e:
            error_message = ""

            if str(e) == "CameraError":
                error_message = "Critical: Camera malfunction."
            elif str(e) == "ObjectDetectionError":
                error_message = "Critical: Object Detection failure."

            logging.critical(error_message)
            await notifier.send_error_message(error_message)

        except Exception as e:
            logging.error(f"Unexpected Error: {e}")



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

    async def test_command(update, context):
        await notifier.send_current_frame(camera)

    try:
        application = ApplicationBuilder().token(bot_token).build()
        bot = application.bot
        notifier = TelegramNotifier(bot_token, default_chat_id)

        detector = CatDetector(model_path)
        global camera
        camera = Camera(detector)

        test_handler = CommandHandler('test', test_command)
        application.add_handler(test_handler)

        loop = asyncio.get_event_loop()
        loop.create_task(periodic_detection(bot, default_chat_id))
        loop.run_until_complete(application.run_polling())
    except Exception as e:
        logging.error(f"Error in main: {e}")


if __name__ == '__main__':
    main()
