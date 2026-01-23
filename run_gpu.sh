#!/bin/bash
# run.sh
# This script launches experiments sequentially while varying parameters defined in rvit.json.
# Usage: ./run_gpu.sh [--gpu GPU_NUMBER]
#   --gpu: GPU number to use (0-7), e.g., ./run_gpu.sh --gpu 4
# If no --gpu specified, uses the device from main_config

# Exit immediately if a command exits with a non-zero status.
#set -e
# Print each command before executing it (for debugging purposes).
#set -x

# Parse GPU argument
GPU_NUM=""
if [[ "$1" == "--gpu" && -n "$2" ]]; then
    GPU_NUM="$2"
    # Validate GPU number is 0-7
    if ! [[ "$GPU_NUM" =~ ^[0-7]$ ]]; then
        echo "Error: GPU number must be between 0 and 7"
        exit 1
    fi
    shift 2
fi

# Parameter arrays

#Backbones
#vit_backbones=("vit_base_patch32_224" "vit_base_patch16_224" "vit_base_patch8_224")
#vit_backbones=("vit_base_patch16_224" "vit_base_patch32_224")
#vit_backbones=("vit_base_patch32_224" "vit_base_patch16_224" "vit_base_patch8_224" "vit_base_patch4_224" "vit_base_patch2_224" )
#patch_size=(4)
vit_backbones=("vit_base_patch32_224")


#implementations=("topx%" "random_topx%" "max_len")
implementations=("random_topx%")
#implementations=("max_len")

mask_thresholds=(1.0)
#mask_thresholds=(0.95 0.9)
#retain_percents=(100 85 50 40 25 15 5)
retain_percents=(25) #15)

# Pretrained weights setting (true/false)
# Set to "true" to use pretrained weights, "false" to train from scratch
# If empty/unset, uses the value from method config file
PRETRAINED="false"

# Fixed configuration files and seed (adjust these paths as needed)
#main_config="configs/main_configs/bigearthnet.json"
main_config="configs/main_configs/mlrsnet.json"
#main_config="configs/main_configs/woody.json"
#main_config="configs/main_configs/flair.json"
#main_config="configs/main_configs/marida.json"
method_config="configs/method/classification/rvit.json"
#method_config="configs/method/segmentation/rvitupernet.json"
#training_config="configs/training/training_per_dataset/rvit/bigearthnet_training.json"
training_config="configs/training/training_per_dataset/rvit/mlrsnet_training.json"
#training_config="configs/training/training_per_dataset/rvitupernet/woody_training.json"
#training_config="configs/training/training_per_dataset/woody_training.json"

seed=222

# Handle GPU configuration
if [ -n "$GPU_NUM" ]; then
    # Extract dataset name from main_config filename (e.g., "mlrsnet" from "configs/main_configs/mlrsnet.json")
    DATASET_NAME=$(basename "$main_config" .json)
    # Create GPU-specific config
    GPU_CONFIG="configs/main_configs/${DATASET_NAME}_gpu${GPU_NUM}.json"
    echo "Using GPU: cuda:${GPU_NUM}"
    
    # Create GPU-specific config file
    python3 << EOF
import pyjson5
import json
import sys

try:
    # Read with pyjson5 to handle comments
    with open("$main_config", 'r') as f:
        config = pyjson5.load(f)
    
    config['device'] = "cuda:${GPU_NUM}"
    
    # Write with standard json (output doesn't need comments)
    with open("$GPU_CONFIG", 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Successfully created GPU config")
except Exception as e:
    print(f"Error creating GPU config: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    
    # Verify file was created
    if [ ! -f "$GPU_CONFIG" ]; then
        echo "Error: Failed to create GPU config file: $GPU_CONFIG"
        exit 1
    fi
    
    main_config="$GPU_CONFIG"
    echo "Created GPU-specific config: $main_config"
fi

# Cleanup function
cleanup() {
    if [ -n "$GPU_NUM" ] && [ -n "$DATASET_NAME" ]; then
        rm -f "configs/main_configs/${DATASET_NAME}_gpu${GPU_NUM}.json"
    fi
}
trap cleanup EXIT INT TERM

# Loop over backbones (outer loop)
for backbone in "${vit_backbones[@]}"; do
  # Loop over implementations (second level)
  for impl in "${implementations[@]}"; do
    if [[ "$impl" == "max_len" || "$impl" == "median" ]]; then
      # For "max_len", set attention_mask true; for "median", set it false.
      if [[ "$impl" == "max_len" ]]; then
        attn_mask=true
      else
        attn_mask=false
      fi
      for mask_threshold in "${mask_thresholds[@]}"; do
        echo "Running experiment with backbone=$backbone, implementation=$impl, mask_threshold=$mask_threshold, attention_mask=$attn_mask, seed=$seed"
        python main.py --config="$main_config" \
                       --seed="$seed" \
                       --method="$method_config" \
                       --training_config="$training_config" \
                       --backbone="$backbone" \
                       --implementation="$impl" \
                       --mask_threshold="$mask_threshold" \
                       --attention_mask="$attn_mask" \
                       ${PRETRAINED:+--pretrained="$PRETRAINED"} \
                       #--patch_size="$patch_size"
        # If python returns a non-zero exit code, the script will exit (due to set -e)
        wait  # Ensures only one script runs at a time before continuing
      done
    elif [[ "$impl" == "topx%" || "$impl" == "random_topx%" ]]; then
      attn_mask=False
      for retain_percent in "${retain_percents[@]}"; do
        echo "Running experiment with backbone=$backbone, implementation=$impl, retain_percent=$retain_percent, attention_mask=$attn_mask, seed=$seed"
        python main.py --config="$main_config" \
                       --seed="$seed" \
                       --method="$method_config" \
                       --training_config="$training_config" \
                       --backbone="$backbone" \
                       --implementation="$impl" \
                       --retain_percent="$retain_percent" \
                       --attention_mask="$attn_mask" \
                       ${PRETRAINED:+--pretrained="$PRETRAINED"} \
                       #--patch_size="$patch_size"
        wait
      done
    elif [[ "$impl" == "vit" ]]; then
      attn_mask=False
      echo "Running experiment with backbone=$backbone, implementation=$impl, retain_percent=$retain_percent, attention_mask=$attn_mask, seed=$seed"
      python main.py --config="$main_config" \
                       --seed="$seed" \
                       --method="$method_config" \
                       --training_config="$training_config" \
                       --backbone="$backbone" \
                       --implementation="$impl" \
                       --attention_mask="$attn_mask" \
                       ${PRETRAINED:+--pretrained="$PRETRAINED"} \
                       #--patch_size="$patch_size"
      wait
    fi
  done
done

echo "End of experiments. Thank you for your patience!"