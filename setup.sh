#!/bin/bash
set -e

echo "Setting up mil environment..."

# Create virtual environment with Python 3.9
uv venv --python 3.9

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies from req.txt
uv pip install -r req.txt

echo "Setup complete! Activate with: source .venv/bin/activate"
