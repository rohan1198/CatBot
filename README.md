# CatBot


### Backstory
Meet CatBot - a little project that turned into a big adventure! It all started with a friendly neighborhood cat that made my doorstep. This furry friend would pop by daily for some cozy time, especially during the cold winter months. Problem was, with me working away in my office, I often missed its visits, and leaving the door open in winter wasn't exactly ideal. Enter CatBot! This nifty system now lets me know when my four-legged buddy is waiting outside.

This journey wasn't just about keeping my feline friend comfy. It was also a deep dive into the world of edge computing and efficient deployment of object detection models. And hey, why not add a dash of convenience with Telegram notifications?

---

### Description
CatBot is a Raspberry Pi-based cat detection system. It uses a camera module for capturing images and YOLOv8 model to detect our feline visitor. When the cat is at the door, I get a notification on Telegram. No more missed hangouts!

---

### Features
- Real-time cat detection using a camera module.
- Telegram notifications with images upon detection.
- Optimized for Raspberry Pi 5 performance.

---

### Requirements
Raspberry Pi 5
Camera Module
Python 3.x
Telegram Bot API Token

---

### Setup

To Do

```
sudo apt update && apt list --upgradable
sudo apt upgrade -y && sudo apt dist-upgrade -y

sudo apt autoremove
sudo apt autoclean

sudo apt install python3 python3-pip python3-dev python3-venv
sudo apt install libopenblas-dev
sudo apt install libatlas-base-dev

mkdir venvs
cd venvs

python3 -m venv yolocat --system-site-packages

source ~/venvs/yolocat/bin/activate

pip install --upgrade pip
pip install --upgrade numpy

pip install python-telegram-bot
pip install ultralytics
pip install python3-dotenv
```

Setup Steps
Clone the Repository
Clone this repository to your Raspberry Pi.

sh
Copy code
git clone https://github.com/your-repo/cat_detection.git
cd cat_detection
Install Required Libraries
Install necessary Python libraries using pip.

sh
Copy code
pip install -r requirements.txt
The requirements.txt should include:

Copy code
numpy
opencv-python
ultralytics
python-telegram-bot
Environment Configuration
Set up an environment file .env in the project root with the following variables:

env
Copy code
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
MODEL_PATH=path_to_your_model
Run the Script on Boot
Set up the script to run on boot using either crontab or systemd.

Using Crontab:

sh
Copy code
crontab -e
Add the following line:

bash
Copy code
@reboot /bin/bash /home/raspi/cat_detection/cat_detector.sh
