#!/bin/bash

# Create a virtual environment if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install dependencies from requirements.txt
python -m pip install -r requirements.txt

# Print a message indicating successful setup
echo "Setup complete. Virtual environment activated."
