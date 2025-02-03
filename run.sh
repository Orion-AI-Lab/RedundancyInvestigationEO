#!/bin/bash
# run.sh
# This script launches experiments concurrently while varying parameters defined in rvit.json.
# Outer loop: backbone, then implementation.
# For "max_len" and "median" implementations, the inner loop varies mask_threshold and sets attention_mask true (only for "max_len").
# For "topx%" implementation, the inner loop varies retain_percent and sets attention_mask false.

# Parameter arrays
vit_backbones=("vit_base_patch16_224.augreg_in21k" "vit_small_patch16_224.augreg_in21k" "vit_tiny_patch16_224.augreg_in21k")
implementations=("max_len" "median" "topx%")
mask_thresholds=(0.25 0.6 0.7 0.75 0.8 0.85)
retain_percents=(85 50 40 25 15 5)

# Fixed configuration files and seed (adjust these paths as needed)
main_config="configs/main_configs/bigearthnet.json"
method_config="configs/method/classification/rvit.json"
training_config="configs/training/training_per_dataset/rvit/bigearthnet_training.json"
seed=222

# Loop over backbones (outer loop)
for backbone in "${vit_backbones[@]}"; do
  # Loop over implementations (second level)
  for impl in "${implementations[@]}"; do
    # If implementation is "max_len" or "median", inner loop varies mask_threshold.
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
                       --attention_mask="$attn_mask" &
      done
    # If implementation is "topx%", inner loop varies retain_percent and attention_mask is false.
    elif [[ "$impl" == "topx%" ]]; then
      attn_mask=false
      for retain_percent in "${retain_percents[@]}"; do
        echo "Running experiment with backbone=$backbone, implementation=$impl, retain_percent=$retain_percent, attention_mask=$attn_mask, seed=$seed"
        python main.py --config="$main_config" \
                       --seed="$seed" \
                       --method="$method_config" \
                       --training_config="$training_config" \
                       --backbone="$backbone" \
                       --implementation="$impl" \
                       --retain_percent="$retain_percent" \
                       --attention_mask="$attn_mask" &
      done
    fi
  done
done

# Wait for all background processes to finish
wait