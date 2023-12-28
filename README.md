# CatBot
Edge computer vision to detect a cat and send a notification to telegram

- Flash the sd card with Raspberry Pi OS (Legacy 32-bit) Full


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