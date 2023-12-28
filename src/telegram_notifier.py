import cv2
import telegram
import logging
import numpy as np
from typing import Optional


class TelegramNotifier(object):
    """
    A class to handle Telegram notifications.

    Attributes:
        bot (telegram.Bot): Instance of the Telegram Bot.
        chat_id (str): Chat ID for sending messages.
    """

    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize the TelegramNotifier with bot token and chat ID.

        Args:
            bot_token (str): Telegram bot token.
            chat_id (str): Telegram chat ID.
        """
        self.bot = telegram.Bot(token=bot_token)
        self.chat_id = chat_id


    def send_message(self, message: str):
        """
        Send a text message via Telegram.

        Args:
            message (str): The message text to send.
        """
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            logging.error(f"Error sending message: {e}")


    def send_photo(self, image: np.ndarray, caption: Optional[str] = None):
        """
        Send a photo via Telegram.

        Args:
            image (np.ndarray): The image to send.
            caption (Optional[str]): The caption for the photo.
        """
        try:
            _, img_encoded = cv2.imencode(".jpg", image)
            self.bot.send_photo(chat_id=self.chat_id, photo=img_encoded.tobytes(), caption=caption)
        except Exception as e:
            logging.error(f"Error sending photo: {e}")
