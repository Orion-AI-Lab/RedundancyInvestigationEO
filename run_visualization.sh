python visualize_masking.py --task classification \
  --checkpoint_path "/mnt/bucket/tpapazaf/RedundancyInvestigation/Checkpoints/MLRSNetCheckpoints/classification/mlrsnet/vit_base_patch16_224/snowy-gorge-140/best_model_state_dict.pt" \
  --config configs/main_configs/mlrsnet.json \
  --method_config configs/method/classification/rvit.json \
  --training_config configs/training/training_per_dataset/rvit/mlrsnet_training.json \
  --webdataset_split test \
  --sample_index 7 \
  --output_path out_cls.png