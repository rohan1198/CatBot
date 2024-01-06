import cv2
import numpy as np
import pytest
from unittest.mock import Mock, patch
from src.telegram_notifier import TelegramNotifier


@pytest.fixture
def mock_bot():
    with patch('telegram.Bot') as mock:
        yield mock


@pytest.fixture
def telegram_notifier(mock_bot):
    return TelegramNotifier(bot_token="dummy_token", chat_id="dummy_chat_id")


def test_send_message(telegram_notifier, mock_bot):
    message = "Test message"
    telegram_notifier.send_message(message)
    mock_bot.return_value.send_message.assert_called_once_with(
        chat_id="dummy_chat_id", text=message)


def test_send_photo(telegram_notifier, mock_bot):
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    caption = "Test photo"

    with patch('cv2.imencode') as mock_imencode:
        mock_imencode.return_value = (True, np.array([1, 2, 3]))
        telegram_notifier.send_photo(image, caption)
        mock_imencode.assert_called_once_with('.jpg', image)
        mock_bot.return_value.send_photo.assert_called_once_with(
            chat_id="dummy_chat_id", photo=np.array([1, 2, 3]).tobytes(), caption=caption)
