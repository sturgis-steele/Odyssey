#!/bin/bash
# ================================================================
# Odyssey Voice Assistant - Full Reproducible Setup Script
# Run this on a fresh Raspberry Pi 5 to recreate the entire environment
# ================================================================

set -e  # Exit immediately if any command fails

echo "=== Updating system packages ==="
sudo apt update && sudo apt upgrade -y

echo "=== Installing system dependencies ==="
sudo apt install -y \
    git build-essential cmake \
    python3-dev python3-venv python3-pip \
    libopenblas-dev portaudio19-dev \
    sox alsa-utils \
    libasound2-plugins

echo "=== Creating project directories ==="
mkdir -p models vosk-model llama-ccp

echo "=== Setting up Python virtual environment ==="
python3 -m venv odyssey-env
source odyssey-env/bin/activate

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Setup complete! ==="
echo "Next steps:"
echo "  1. Download your GGUF models into the 'models/' folder (see models/README.md)"
echo "  2. Place your Vosk model in 'vosk-model/model/'"
echo "  3. Run: cd llama-ccp && python computer_llama.py"
