import cv2
import logging
import telegram
import numpy as np
import asyncio
from typing import Optional
from src.camera import Camera


class TelegramNotifier:
    """
    Class to handle Telegram notifications.

    Attributes:
        bot (telegram.Bot): Telegram bot instance.
        bhat_id (str): The chat ID for sending notifications.
    """
    def __init__(self, bot_token: str, chat_id: str):
        try:
            self.bot = telegram.Bot(token=bot_token)
            self.chat_id = chat_id
        except Exception as e:
            logging.error(f"TelegramNotifier initialization failed: {e}")
            raise


    def send_message(self, message: str):
        """
        Sends a text message via Telegram.

        Args:
            message (str): The message to send.
        """
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            logging.error(f"Error sending message: {e}")


    def send_photo(self, image: np.ndarray, caption: Optional[str] = None):
        """
        Sends a photo via Telegram.

        Args:
            image (np.ndarray): The image to send.
            caption (Optional[str]): The caption for the photo.
        """
        try:
            _, img_encoded = cv2.imencode(".jpg", image)
            self.bot.send_photo(chat_id=self.chat_id, photo=img_encoded.tobytes(), caption=caption)
        except Exception as e:
            logging.error(f"Error sending photo: {e}")


    async def send_error_message(self, error_message: str):
        """
        Sends an error message via Telegram.

        Args:
            error_message (str): The error message to send.
        """
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=error_message)
        except Exception as e:
            logging.error(f"Error sending error message via Telegram: {e}")


    async def send_current_frame(self, camera: Camera):
        """
        Captures the current frame from the camera and sends it as a photo.
        Meant to be a test function.

        Args:
            camera (Camera): The camera object to capture the frame.
        """
        try:
            img = camera.capture_frame()
            await asyncio.get_running_loop().run_in_executor(None, lambda: self.send_photo(img, "Current Frame"))
        except Exception as e:
            logging.error(f"Error in capturing/sending current frame: {e}")
            await self.send_message("Error: Unable to capture and send current frame")
