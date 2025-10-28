#!/bin/bash
# Quick Start Script for Phi-3 Payments Fine-tuning
# Run this to set everything up automatically

set -e  # Exit on error

echo "=================================================="
echo "Phi-3 Payments Fine-tuning - Quick Start"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

echo "✓ Python 3 detected: $(python3 --version)"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. Make sure CUDA is installed."
    echo "   You can continue but training may not work without a GPU."
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ CUDA detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "✓ All dependencies installed"

# Generate dataset
echo ""
echo "=================================================="
echo "Step 1: Generating synthetic payments dataset"
echo "=================================================="
python generate_payments_dataset.py

# Ask if user wants to start training
echo ""
echo "=================================================="
echo "Step 2: Fine-tuning (takes 30-45 minutes)"
echo "=================================================="
echo ""
read -p "Start training now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting training..."
    echo "You can monitor GPU usage in another terminal with: watch -n 1 nvidia-smi"
    echo ""
    python finetune_phi3_payments.py
    
    echo ""
    echo "=================================================="
    echo "Training complete! Testing the model..."
    echo "=================================================="
    python test_payments_model.py
    
    echo ""
    echo "✓ All done!"
    echo ""
    echo "Next steps:"
    echo "  - Try interactive mode: python test_payments_model.py interactive"
    echo "  - Customize dataset: edit generate_payments_dataset.py"
    echo "  - Adjust training: edit finetune_phi3_payments.py"
else
    echo ""
    echo "Skipping training. You can run it later with:"
    echo "  source venv/bin/activate"
    echo "  python finetune_phi3_payments.py"
fi

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
