#!/bin/bash
set -e

echo "--= Create Display =--"

display="90"

echo "Starting Xvfb :$display..."
Xvfb :$display -screen 0 1024x1024x24 -ac +extension GLX +render -noreset &

DISPLAY=":$display"

# Run a tiny initialization to force Unity to start and create Player.log
echo "--= Running AI2Thor initialization... =--"
python Magmathor/Model/force_ai2thor_initialization.py || true
echo "--= Create Display Done =--"
