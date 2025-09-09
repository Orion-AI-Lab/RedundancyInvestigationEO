#!/bin/bash
# run.sh
# This script launches experiments sequentially while varying parameters defined in rvit.json.
# It stops execution if any experiment returns a nonzero exit code.
# Debugging options are enabled for detailed execution tracing.

# Exit immediately if a command exits with a non-zero status.
#set -e
# Print each command before executing it (for debugging purposes).
#set -x

# Parameter arrays
#CLASSIFICATION
#vit_backbones=("vit_tiny_patch16_224.augreg_in21k" "vit_small_patch16_224.augreg_in21k" "vit_base_patch16_224.augreg_in21k")
vit_backbones=("vit_small_patch8_224")


#SEGMENTATION
#vit_backbones=("vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224")
#vit_backbones=("vit_tiny_patch16_224")
#ekkremei max_len me small kai ta katw


#implementations=("topx%" "random_topx%" "median" "max_len")
implementations=("topx%")
#implementations=("median")

mask_thresholds=(0.85 0.8 0.75 0.7 0.6 0.25)
#mask_thresholds=(0.8)
#retain_percents=(85 50 40 25 15 5)
#retain_percents=(100 85 50 40 25 15 5)
retain_percents=(100)

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
                       --attention_mask="$attn_mask"
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
                       --attention_mask="$attn_mask"
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
                       --attention_mask="$attn_mask"
      wait
    fi
  done
done

echo "All experiments completed successfully."
