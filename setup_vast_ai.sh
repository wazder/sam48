#!/bin/bash

# Setup script for vast.ai RTX A4000 environment
# Run this script after connecting to your vast.ai instance

set -e  # Exit on any error

echo "🚀 Setting up Sam48 on vast.ai RTX A4000"
echo "========================================"

# Update system
echo "📦 Updating system packages..."
apt-get update -q
apt-get install -y git wget curl unzip ffmpeg

# Check Python version
echo "🐍 Checking Python version..."
python3 --version
pip3 --version

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p test_videos
mkdir -p test_output
mkdir -p output/{logs,masks,videos}

# Download SAM model if not present
SAM_MODEL="sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_MODEL" ]; then
    echo "🔽 Downloading SAM model..."
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    echo "✅ SAM model downloaded"
else
    echo "✅ SAM model already exists"
fi

# Download YOLO model (will be done automatically by ultralytics)
echo "🎯 YOLO model will be downloaded automatically on first run"

# Make scripts executable
chmod +x download_test_video.py
chmod +x test_vast_ai.py

# Check GPU availability
echo "🖥️  Checking GPU..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run the test suite: python3 test_vast_ai.py"
echo "2. Or download video manually: python3 download_test_video.py"
echo "3. Or run pipeline directly: python3 main.py test_videos/test_video.mp4"
echo ""
echo "Happy testing! 🎉"