#!/bin/sh

sed -i "s|PORT=8888|PORT=${PORT:-8888}|" .env
sed -i "s|LOG_LEVEL=INFO|LOG_LEVEL=${LOG_LEVEL:-INFO}|" .env

pip uninstall -y numpy
pip install numpy==1.15.4

python websocket-jambonz-server.py
