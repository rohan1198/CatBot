import cv2
import telegram
import numpy as np
from typing import Optional


class TelegramNotifier(object):
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = telegram.Bot(token=bot_token)
        self.chat_id = chat_id

    
    def send_message(self, message: str):
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            print(f"Error sending message: {e}")

    
    def send_photo(self, image: np.ndarray, caption: Optional[str] = None):
        try:
            _, img_encoded = cv2.imencode(".jpg", image)
            self.bot.send_photo(chat_id=self.chat_id, photo=img_encoded.tobytes(), caption=caption)
        except Exception as e:
            print(f"Error sending photo: {e}")
