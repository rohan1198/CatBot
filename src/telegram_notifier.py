import cv2
import logging
import telegram
import numpy as np
from typing import Optional


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
