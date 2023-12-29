#!/bin/bash

# Configuration
MAX_RETRIES=3
RETRY_COUNT=0
SCRIPT_PATH="/home/raspi/CatBot/main.py"
LOG_FILE="/home/raspi/CatBot/logfile.log"
VENV_PATH="/home/raspi/venvs/yolocat/bin/activate"

echo "Starting script monitoring" >> "$LOG_FILE"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if pgrep -f $SCRIPT_PATH > /dev/null
    then
        echo "$(date): Script is running." >> "$LOG_FILE"
    else
        echo "$(date): Script not running, starting script." >> "$LOG_FILE"
        # Activate the virtual environment
        source $VENV_PATH
        # Run the script in the virtual environment
        python3 $SCRIPT_PATH &
        RETRY_COUNT=$((RETRY_COUNT+1))
    fi
    sleep 60  # Check every 60 seconds
done

echo "$(date): Maximum retries reached, rebooting system." >> "$LOG_FILE"
sudo reboot
