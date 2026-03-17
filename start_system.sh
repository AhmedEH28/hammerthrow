#!/bin/bash

# Hammer Throw Analysis System Startup Script
echo "=========================================="
echo "Hammer Throw Analysis System"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p uploads
mkdir -p static/detections
mkdir -p results
mkdir -p models
mkdir -p test_throws

# Check if models exist
if [ ! "$(ls -A models/)" ]; then
    echo "Warning: No models found in models/ directory"
    echo "Please place your YOLO model files (.pt) in the models/ directory"
fi

# Check if test_throws directory has any content
if [ ! "$(ls -A test_throws/)" ]; then
    echo "Info: No test throws found in test_throws/ directory"
    echo "You can upload test throws using the web interface"
fi

echo ""
echo "Starting Hammer Throw Analysis System..."
echo "=========================================="
echo ""
echo "🌟 ADVANCED INTERFACE (DEFAULT):   http://localhost:5000/"
echo "🌟 ADVANCED INTERFACE (ALT):       http://localhost:5000/advanced"
echo "🔧 SIMPLIFIED INTERFACE:           http://localhost:5000/simple"
echo ""
echo "📁 To add test throws manually, place them in: ./test_throws/"
echo "📝 Each test throw folder should contain:"
echo "   - side_view.mp4 (or similar with 'side' in name)"
echo "   - back_view.mp4 (or similar with 'back' in name)" 
echo "   - calibration.csv (calibration data)"
echo "   - metadata.json (optional: release point, frame rate)"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start the Flask application
python3 app.py
