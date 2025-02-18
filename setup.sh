#!/bin/bash

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Force install huggingface-hub and tf-explain AGAIN - to ensure they are installed
pip install huggingface-hub
pip install tf-explain

echo "Dependencies installed from requirements.txt and force-installed huggingface-hub and tf-explain."
