#!/bin/bash
# Phase 6 Execution Script
# Includes Data Cleaning, and Training with FGM, Mixup, and Codec Augmentation

set -e # Exit on error

# 1. Data Cleaning (Step 1)
# ------------------------------------------------------------------
echo "Step 1: Running Data Cleaning (Filtering top 2% dirty samples)..."
# NOTE: Data cleaning with WavLM-Large is NOT recommended on CPU (may be OOM-killed).
# Require CUDA unless user explicitly forces CPU.
python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: CUDA not available. Please enable GPU before running data cleaning."; exit 1; }

python filter_dirty_data.py \
    --config config/Phase5_Finetune.conf \
    --model_path exp_result/LA_Phase5_Finetune_ep20_bs12/weights/best.pth \
    --output_path dirty_samples_phase5.txt \
    --filter_ratio 0.02 \
    --batch_size 8 \
    --device cuda \
    --amp

# Check if cleaning succeeded
if [ ! -f "dirty_samples_phase5_cleaned_protocol.txt" ]; then
    echo "Error: Data cleaning failed. cleaned protocol file not found."
    exit 1
fi

echo "Data Cleaning Complete. Cleaned protocol: dirty_samples_phase5_cleaned_protocol.txt"


# 2. Phase 6 Training (Steps 2, 3, 4)
# ------------------------------------------------------------------
echo "Step 2: Starting Phase 6 Training (FGM + Mixup + Codec Augmentation)..."

# Ensure the config points to the cleaned protocol
# We inject the custom protocol path into the config dynamically or use the one we prepared
# Phase6_Proposed.conf doesn't have the path hardcoded, so we will use a temporary config or 
# pass it via command line argument if we supported it.
# BUT, main.py checks config["data_config"]["custom_train_protocol"].
# We need to add this to the config file or create a variant.

# Let's create a runtime config
cp config/Phase6_Proposed.conf config/Phase6_Run.conf

# Use python to insert the path into the JSON/HOCON-like config structure?
# Our config is JSON.
# We can just append the line using sed if we are careful, or use a python one-liner.
# Config format:
# "data_config": { ... }
# We want to add "custom_train_protocol": "./dirty_samples_phase5_cleaned_protocol.txt" inside data_config.

# Simple Python script to update config
python -c "
import json
with open('config/Phase6_Run.conf', 'r') as f:
    config = json.load(f)

config['data_config']['custom_train_protocol'] = './dirty_samples_phase5_cleaned_protocol.txt'
# Also ensure Codec Augmentation is ON (Action 4)
config['data_config']['use_codec_aug'] = True

with open('config/Phase6_Run.conf', 'w') as f:
    json.dump(config, f, indent=4)
"

echo "Config updated with cleaned protocol and Codec Augmentation."

# Run Training
python main.py --config config/Phase6_Run.conf

echo "Phase 6 Pipeline Complete."

