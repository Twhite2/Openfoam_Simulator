#!/bin/bash
set -e

# Source OpenFOAM environment
source /opt/openfoam12/etc/bashrc

# Set up display
export DISPLAY=:1
export RESOLUTION=${RESOLUTION:-1920x1080}

# Initialize X server
Xvfb :1 -screen 0 $RESOLUTION -ac &

# Wait for X server to start
sleep 2

# Start VNC server
x11vnc -display :1 -forever -usepw -create &

# Start noVNC
/usr/local/novnc/utils/novnc_proxy --vnc localhost:5900 --listen 6080 &

# Set up OpenFOAM run directory
mkdir -p /app/F3D_visualizer/runs
chmod -R 777 /app/F3D_visualizer/runs

# Print some useful information
echo "==============================================="
echo "F3D_visualizer - Oil & Gas CFD Visualization System"
echo "==============================================="
echo "VNC server running on port 5900"
echo "noVNC web client available at http://localhost:6080/vnc.html"
echo "OpenFOAM version: $(foamVersion 2>/dev/null || echo 'OpenFOAM 12')"
echo "ParaView version: $(paraview --version 2>/dev/null || echo 'Latest from apt')"
echo "==============================================="

# Start supervisord to manage processes if it exists in the arguments
if [ "$1" = "supervisord" ]; then
    exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
# Otherwise, start the application
elif [ "$1" = "app" ]; then
    cd /app/F3D_visualizer
    python3 src/main.py
# Or provide a shell for debugging
elif [ "$1" = "shell" ]; then
    exec /bin/bash
# Default: start supervisord
else
    exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
fi