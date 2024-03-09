#!/bin/bash

[Unit]
Description=Cat Detection system
After=network.target

[Service]
Type=forking
ExecStart=/bin/bash /home/raspi/CatBot/cat_detector.sh start
ExecStop=/bin/bash /home/raspi/CatBot/cat_detector.sh stop
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target