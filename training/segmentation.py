import os
import pprint

import kornia
import numpy as np
import pyjson5 as json
import torch
import tqdm

import utilities.model_utilities as model_utils
import utilities.utils as utils
import wandb
from utilities.model_utilities import adjust_learning_rate, get_current_learning_rate
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

def early_stopping(val_loss, best_loss, counter, patience=5):
    if val_loss < best_loss:
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping")
        return True, counter
    else:
        return False, counter


def train_epoch(train_loader, model, optimizer, criterion, epoch, configs, scaler):
    if not configs["linear_evaluation"] and not configs["fully_finetune"]:
        model.train()
    else:
        model.train()
        modality_dictionary = {v: k for k, v in configs["modality_channels"].items()}
        available_modalities = configs["dataset_modality_index"][configs["dataset"]]
        spectral_keys = []
        desired_indices = []
        for modality in available_modalities:
            spectral_keys.append(int(modality_dictionary[modality]))
            desired_indices.append(available_modalities[modality])

    for idx, batch in enumerate(tqdm.tqdm(train_loader)):
        if "samples_per_epoch" in configs:
            if (idx + 1) * configs["batch_size"] > configs["samples_per_epoch"]:
                break
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=configs["mixed_precision"]):
            image, label, keys = batch
            
            image = image.to(configs["device"])
            label = label.to(configs["device"])

            if not configs["linear_evaluation"] and not configs["fully_finetune"]:
                if configs['architecture']=='rvitupernet':
                    out, mask = model(image, label=label, keys=keys)
                elif configs['architecture']=='unet':   
                    out = model(image)
                else:
                    out = model(image, label=label)
            else:
                image = image[:, desired_indices, :, :]

                out = model((image, spectral_keys))

            label = label.long()
            if configs['architecture']=='rvitupernet':
                # Convert mask to float to maintain computation graph
                # The mask from model is already interpolated to match output size
                mask_float = mask.float()
                loss = F.cross_entropy(out, label, reduction='none')  # Shape: [B, H, W]
                # Ensure mask shape matches loss shape [B, H, W]
                if mask_float.dim() == 2:
                    # If mask is [H, W], expand to [B, H, W]
                    mask_float = mask_float.unsqueeze(0).expand_as(loss)
                elif mask_float.dim() == 3:
                    if mask_float.shape[0] == 1 and loss.shape[0] > 1:
                        # If mask is [1, H, W], expand to [B, H, W]
                        mask_float = mask_float.expand_as(loss)
                    elif mask_float.shape != loss.shape:
                        # Reshape if dimensions don't match
                        mask_float = F.interpolate(
                            mask_float.unsqueeze(1).float(), 
                            size=(loss.shape[1], loss.shape[2]), 
                            mode='nearest'
                        ).squeeze(1)
                masked_loss = loss * mask_float
                # Ensure denominator is a float tensor in the computation graph
                mask_sum = mask_float.sum().clamp(min=1.0)
                loss = masked_loss.sum() / mask_sum
            else:
                loss = criterion(out, label)
            
            # Break after the first 50 batches for testing
            #if idx == 50:
            #    break
                
        if idx % 100 == 0:
            log_dict = {"Epoch": epoch, "Iteration": idx, "train loss": loss.item()}
            if configs["wandb"]:
                wandb.log(log_dict)
            else:
                print(log_dict)
        if configs["mixed_precision"]:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if configs["schedule"] == "cos":
            if "samples_per_epoch" in configs:
                num_steps = configs["samples_per_epoch"] // configs["batch_size"]
            else:
                num_steps = 10000
            adjust_learning_rate(optimizer, idx / num_steps + epoch, configs)


def train(configs):

    print("=" * 20)
    print("Initializing segmentation trainer")
    print("=" * 20)
    
    metrics = utils.initialize_metrics(configs)
    criterion = utils.create_loss(configs)
    base_model = model_utils.create_model(configs)
    optimizer = utils.create_optimizer(configs)(
        base_model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"]
    )

    # compile model (torch 2.0)
    model = base_model  # torch.compile(base_model)

    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    train_loader, val_loader, _ = utils.create_dataloaders(configs)
    model.to(configs["device"])
    best_loss = 10000.0
    early_stop_counter = 0
    for epoch in range(configs["epochs"]):
        print("=" * 20)
        print("Epoch: ", epoch)
        print("-" * 20)
        print("Begin Training")

        train_epoch(train_loader, model, optimizer, criterion, epoch, configs, scaler)

        val_loss = test(configs, phase="val", model=model, criterion=criterion, loader=val_loader, epoch=epoch)
        if "early_stopping" in configs and configs["early_stopping"] > 0:
            early_stop, early_stop_counter = early_stopping(
                val_loss, best_loss, early_stop_counter, patience=configs["early_stopping"]
            )
        else:
            early_stop = False

        if val_loss < best_loss:
            best_loss = val_loss
            print("New best validation loss: ", best_loss)
            print("Saving checkpoint")
            # Store checkpoint
            torch.save(base_model, os.path.join(configs["checkpoint_path"], "best_model.pt"))
            torch.save(base_model.state_dict(), os.path.join(configs["checkpoint_path"], "best_model_state_dict.pt"))
        if early_stop:
            print("Early stopping at epoch: ", epoch)
            break


def test(configs, phase, model=None, loader=None, criterion=None, epoch="Test"):
    if phase == "test":
        print("-" * 20)
        print("Begin Testing")

        _, _, loader = utils.create_dataloaders(configs)
        criterion = utils.create_loss(configs)

        if configs["eval_checkpoint"] is not None:
            print("=" * 20)
            print("Evaluating segmentation for phase: ", phase, " with checkpoint: ")
            print(configs["eval_checkpoint"])
            print("=" * 20)
            # Load model from checkpoint
            # In PyTorch >= 2.6, torch.load defaults to weights_only=True.
            # We explicitly set weights_only=False because this checkpoint
            # was created by our own code and contains the full model object.
            model = torch.load(
                os.path.join(configs["eval_checkpoint"]),
                map_location=configs["device"],
                weights_only=False,
            )
        else:
            # infer checkpoint path from configs
            # Load model from checkpoint
            # In PyTorch >= 2.6, torch.load defaults to weights_only=True.
            # We explicitly set weights_only=False because this checkpoint
            # was created by our own code and contains the full model object.
            model = torch.load(
                os.path.join(configs["checkpoint_path"], "best_model.pt"),
                map_location=configs["device"],
                weights_only=False,
            )

        # compile model (torch 2.0)
        # model = torch.compile(model)
    elif phase == "val":
        print("-" * 20)
        print("Begin Evaluation")

    else:
        print("Uknown phase!")
        exit(3)

    if configs["linear_evaluation"] or configs["fully_finetune"]:
        modality_dictionary = {v: k for k, v in configs["modality_channels"].items()}
        available_modalities = configs["dataset_modality_index"][configs["dataset"]]
        spectral_keys = []
        desired_indices = []
        for modality in available_modalities:
            spectral_keys.append(int(modality_dictionary[modality]))
            desired_indices.append(available_modalities[modality])

    metrics = utils.initialize_metrics(configs)
    model.to(configs["device"])
    model.eval()
    total_loss = 0.0

    # Images to log to wandb
    first_image = None
    first_prediction = None
    first_mask = None
    num_samples = 0
    for idx, batch in enumerate(tqdm.tqdm(loader)):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=configs["mixed_precision"]):
                # Handle batch unpacking - test/val loaders may or may not have keys
                if len(batch) == 3:
                    image, label, keys = batch
                else:
                    image, label = batch
                    keys = None
                
                image = image.to(configs["device"])
                label = label.to(configs["device"])

                if not configs["linear_evaluation"] and not configs["fully_finetune"]:
                    if configs['architecture'] == 'rvitupernet':
                        out = model(image, label=label, keys=keys)
                    elif configs['architecture'] == 'unet':
                        out = model(image)
                    else:
                        out = model(image, label=label)
                else:
                    image = image[:, desired_indices, :, :]
                    out = model((image, spectral_keys))

                if idx == 0:
                    predictions = out.argmax(1)
                    first_image = image.detach().cpu()[0]
                    first_prediction = predictions.detach().cpu()[0]
                    first_mask = label.detach().cpu()[0]
                label = label.long()
                loss = criterion(out, label)
                total_loss += loss.item()
                for metric in metrics:
                    metric(out, label)
                num_samples += image.shape[0]

    total_loss = total_loss / num_samples
    log_dict = {"Epoch": epoch, phase + " loss": total_loss}
    
    # Compute and log both aggregated and per-class metrics
    per_class_metrics = {}  # Store per-class metrics for diagnostic summary
    for idx, metric in enumerate(metrics):
        if metric.average != "none":
            # Log aggregated metrics (micro, macro, weighted)
            metric_value = metric.compute()
            log_dict[phase + " " + metric.average + " " + metric.__class__.__name__] = metric_value
        # COMMENTED OUT: Per-class metrics computation for speed
        # else:
        #     # Always log per-class metrics (not just for test phase)
        #     scores = metric.compute()
        #     metric_name = metric.__class__.__name__
        #     if metric_name not in per_class_metrics:
        #         per_class_metrics[metric_name] = {}
        #     for class_idx in range(scores.shape[0]):
        #         score_value = scores[class_idx].item()
        #         log_dict[phase + " " + metric_name + " Class: " + str(class_idx)] = score_value
        #         per_class_metrics[metric_name][class_idx] = score_value
    
    # COMMENTED OUT: Print diagnostic summary for per-class metrics (for speed)
    # if per_class_metrics and phase == "val":
    #     print("\n" + "="*60)
    #     print(f"PER-CLASS METRICS SUMMARY ({phase.upper()})")
    #     print("="*60)
    #     for metric_name, class_scores in per_class_metrics.items():
    #         print(f"\n{metric_name} per class:")
    #         for class_idx in sorted(class_scores.keys()):
    #             score = class_scores[class_idx]
    #             print(f"  Class {class_idx}: {score:.4f}")
    #         
    #         # Check for significant class imbalance in performance
    #         if len(class_scores) > 1:
    #             scores_list = list(class_scores.values())
    #             max_score = max(scores_list)
    #             min_score = min(scores_list)
    #             score_range = max_score - min_score
    #             if score_range > 0.3:  # If difference > 30%
    #                 print(f"  ⚠️  WARNING: Large performance gap detected!")
    #                 print(f"     Best class: {max_score:.4f}, Worst class: {min_score:.4f}, Range: {score_range:.4f}")
    #                 print(f"     This suggests class imbalance. Consider using weighted loss.")
    #     
    #     # Check macro vs micro F1-score gap (strong indicator of class imbalance)
    #     micro_f1 = log_dict.get(f"{phase} micro F1Score", None)
    #     macro_f1 = log_dict.get(f"{phase} macro F1Score", None)
    #     if micro_f1 is not None and macro_f1 is not None:
    #         gap = micro_f1 - macro_f1
    #         print(f"\n📊 F1-Score Analysis:")
    #         print(f"   Micro F1: {micro_f1:.4f} (weighted by class frequency)")
    #         print(f"   Macro F1: {macro_f1:.4f} (equal weight per class)")
    #         print(f"   Gap: {gap:.4f}")
    #         if gap > 0.3:
    #             print(f"   ⚠️  CRITICAL: Very large gap ({gap:.4f}) indicates SEVERE class imbalance!")
    #             print(f"      The model is heavily biased toward dominant classes.")
    #             print(f"      Your model is likely predicting mostly the dominant class.")
    #             print(f"      ACTION REQUIRED: Implement weighted loss immediately!")
    #         elif gap > 0.2:
    #             print(f"   ⚠️  CRITICAL: Large gap ({gap:.4f}) indicates severe class imbalance!")
    #             print(f"      The model is biased toward dominant classes.")
    #             print(f"      RECOMMENDATION: Use weighted loss or class-balanced sampling.")
    #         elif gap > 0.1:
    #             print(f"   ⚠️  WARNING: Moderate gap ({gap:.4f}) suggests class imbalance.")
    #     
    #     # Verify segmentation correctness using precision and recall
    #     print(f"\n✅ Segmentation Correctness Verification:")
    #     precision_per_class = per_class_metrics.get("Precision", {})
    #     recall_per_class = per_class_metrics.get("Recall", {})
    #     
    #     if precision_per_class and recall_per_class:
    #         print(f"   Per-class Precision (when model predicts a class, is it correct?):")
    #         all_precisions_valid = True
    #         for class_idx in sorted(precision_per_class.keys()):
    #             prec = precision_per_class[class_idx]
    #             print(f"     Class {class_idx}: {prec:.4f}", end="")
    #             if prec < 0.5:
    #                 print(" ⚠️  LOW - Model makes many false positives for this class")
    #                 all_precisions_valid = False
    #             elif prec > 0.8:
    #                 print(" ✓ Good")
    #             else:
    #                 print(" - Acceptable")
    #         
    #         print(f"\n   Per-class Recall (when a class exists, does model find it?):")
    #         all_recalls_valid = True
    #         for class_idx in sorted(recall_per_class.keys()):
    #             rec = recall_per_class[class_idx]
    #             print(f"     Class {class_idx}: {rec:.4f}", end="")
    #             if rec < 0.3:
    #                 print(" ⚠️  VERY LOW - Model is missing this class")
    #                 all_recalls_valid = False
    #             elif rec < 0.5:
    #                 print(" ⚠️  LOW - Model misses many instances of this class")
    #                 all_recalls_valid = False
    #             elif rec > 0.7:
    #                 print(" ✓ Good")
    #             else:
    #                 print(" - Acceptable")
    #         
    #         # Check if model is just predicting one class
    #         iou_per_class = per_class_metrics.get("JaccardIndex", {})
    #         if iou_per_class:
    #             non_zero_classes = [c for c, score in iou_per_class.items() if score > 0.01]
    #             if len(non_zero_classes) < configs["num_classes"]:
    #                 missing_classes = set(range(configs["num_classes"])) - set(non_zero_classes)
    #                 print(f"\n   ⚠️  WARNING: Model is NOT predicting classes: {sorted(missing_classes)}")
    #                 print(f"      IoU < 0.01 for these classes suggests the model ignores them.")
    #             else:
    #                 print(f"\n   ✓ Model predicts all {configs['num_classes']} classes")
    #         
    #         if all_precisions_valid and all_recalls_valid:
    #             print(f"\n   ✓ Overall: Segmentation appears to be performed correctly")
    #             print(f"      All classes have reasonable precision and recall.")
    #         else:
    #             print(f"\n   ⚠️  Some classes have low precision/recall - segmentation may be incomplete")
    #     else:
    #         print(f"   (Precision/Recall metrics not available - add 'precision' and 'recall' to metrics config)")
    #     
    #     print("="*60 + "\n")
    
    # Analyze class distribution and model bias for the first sample
    # if first_mask is not None and first_prediction is not None:
    #     # Compute class distribution in ground truth
    #     gt_np = first_mask.detach().cpu().numpy()
    #     pred_np = first_prediction.detach().cpu().numpy()
        
    #     # Count pixels per class in ground truth
    #     unique_gt, counts_gt = np.unique(gt_np, return_counts=True)
    #     gt_class_dist = {int(cls): int(count) for cls, count in zip(unique_gt, counts_gt)}
    #     total_pixels_gt = gt_np.size
        
    #     # Count pixels per class in prediction
    #     unique_pred, counts_pred = np.unique(pred_np, return_counts=True)
    #     pred_class_dist = {int(cls): int(count) for cls, count in zip(unique_pred, counts_pred)}
    #     total_pixels_pred = pred_np.size
        
    #     # Log class distribution for first sample (for visualization/debugging)
    #     for class_idx in range(configs["num_classes"]):
    #         gt_pct = (gt_class_dist.get(class_idx, 0) / total_pixels_gt * 100) if total_pixels_gt > 0 else 0
    #         pred_pct = (pred_class_dist.get(class_idx, 0) / total_pixels_pred * 100) if total_pixels_pred > 0 else 0
    #         log_dict[f"{phase}_sample_class_{class_idx}_gt_pct"] = gt_pct
    #         log_dict[f"{phase}_sample_class_{class_idx}_pred_pct"] = pred_pct
        
    #     # Check for class imbalance warning in this sample
    #     if len(gt_class_dist) > 0:
    #         max_class = max(gt_class_dist.items(), key=lambda x: x[1])
    #         max_class_pct = (max_class[1] / total_pixels_gt * 100) if total_pixels_gt > 0 else 0
    #         if max_class_pct > 70:  # If dominant class is >70% of pixels
    #             print(f"⚠️  WARNING: In visualized sample, Class {max_class[0]} dominates {max_class_pct:.1f}% of ground truth pixels")
    #             print(f"   Model predicted {pred_class_dist.get(max_class[0], 0) / total_pixels_pred * 100:.1f}% as Class {max_class[0]}")
    #             print(f"   Consider using weighted loss or class-balanced sampling if this pattern persists across the dataset.")
        
    #     # Verify model is actually segmenting (not just predicting one class)
    #     predicted_classes = set(pred_class_dist.keys())
    #     gt_classes = set(gt_class_dist.keys())
    #     if len(predicted_classes) == 1 and len(gt_classes) > 1:
    #         print(f"⚠️  CRITICAL: Model is predicting ONLY class {list(predicted_classes)[0]} in this sample!")
    #         print(f"   Ground truth has classes: {sorted(gt_classes)}")
    #         print(f"   This indicates the model is NOT performing segmentation correctly.")
    #     elif len(predicted_classes) < len(gt_classes):
    #         missing = gt_classes - predicted_classes
    #         print(f"⚠️  WARNING: Model is NOT predicting classes {sorted(missing)} in this sample")
    #         print(f"   Ground truth has: {sorted(gt_classes)}, Model predicted: {sorted(predicted_classes)}")
    if configs["wandb"]:
        # Log metrics to wandb
        wandb.log(log_dict)
        
        # Log wandb mask overlay visualization (interactive masks) if enabled
        if configs.get("log_images", False) and first_image is not None and first_mask is not None and first_prediction is not None:
            try:
                class_labels = {}
                for i in list(range(configs["num_classes"])):
                    class_labels[i] = str(i)
                
                # Prepare image for wandb mask overlay
                if configs["normalization"] == "standard":
                    img_for_mask = first_image[:3, :, :].unsqueeze(0)  # [1, 3, H, W]
                    denorm_mean = torch.tensor(configs["mean"])[:3] if len(configs["mean"]) >= 3 else torch.tensor(configs["mean"])
                    denorm_std = torch.tensor(configs["std"])[:3] if len(configs["std"]) >= 3 else torch.tensor(configs["std"])
                    denormalizer = kornia.enhance.Denormalize(denorm_mean, denorm_std)
                    img_for_mask = denormalizer(img_for_mask).squeeze(0)  # [3, H, W]
                    img_for_mask = img_for_mask.permute(1, 2, 0)  # [H, W, 3]
                    # Normalize to [0, 255]
                    img_min = img_for_mask.min()
                    img_max = img_for_mask.max()
                    if img_max > img_min:
                        img_for_mask = (img_for_mask - img_min) / (img_max - img_min)
                    img_for_mask = (img_for_mask * 255).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
                else:
                    img_for_mask = first_image[:3, :, :].permute(1, 2, 0).detach().cpu().numpy()
                    if img_for_mask.max() <= 1.0:
                        img_for_mask = (img_for_mask * 255).astype(np.uint8)
                    else:
                        img_for_mask = img_for_mask.clamp(0, 255).astype(np.uint8)
                
                # Create wandb Image with mask overlays (interactive in wandb UI)
                mask_img = wandb.Image(
                    img_for_mask,
                    masks={
                        "predictions": {
                            "mask_data": first_prediction.detach().cpu().numpy(),
                            "class_labels": class_labels
                        },
                        "ground_truth": {
                            "mask_data": first_mask.detach().cpu().numpy(),
                            "class_labels": class_labels
                        },
                    },
                )
                wandb.log({f"{phase}_sample_with_masks": mask_img})
            except Exception as e:
                print(f"Warning: Could not generate wandb mask overlay due to: {e}")
    else:
        print(log_dict)

    # ---------------------------
    # Visualization: Side-by-side comparison of raw image, ground truth, and prediction
    if configs.get("log_images", False) and first_image is not None and first_mask is not None and first_prediction is not None:
        try:
            # Ensure first_image is in [C, H, W] format
            # Check if image is already in [C, H, W] format (channels first)
            if len(first_image.shape) == 3:
                # If first dimension is large (likely height/width), assume [H, W, C] and permute
                # If first dimension is small (likely channels), assume it's already [C, H, W]
                if first_image.shape[0] > 100:  # Height/width is typically > 100, channels are typically < 20
                    # Shape is likely [H, W, C], permute to [C, H, W]
                    first_image = first_image.permute(2, 0, 1)
                # Otherwise, assume it's already [C, H, W] format

            # Prepare raw image for visualization (first 3 channels, denormalized if needed)
            if configs["normalization"] == "standard":
                # Extract only the first three values from the mean and std
                denorm_mean = torch.tensor(configs["mean"])[:3] if len(configs["mean"]) >= 3 else torch.tensor(configs["mean"])
                denorm_std = torch.tensor(configs["std"])[:3] if len(configs["std"]) >= 3 else torch.tensor(configs["std"])
                # Extract the first three channels from the image for visualization
                img_to_denorm = first_image[:3, :, :].unsqueeze(0)  # shape: [1, 3, H, W]
                denormalizer = kornia.enhance.Denormalize(denorm_mean, denorm_std)
                first_img_denorm = denormalizer(img_to_denorm).squeeze(0)
                # Permute to H, W, C for display and scale to [0, 255]
                first_img_disp = first_img_denorm.permute(1, 2, 0)
                # Normalize to [0, 1] then scale to [0, 255]
                img_min = first_img_disp.min()
                img_max = first_img_disp.max()
                if img_max > img_min:
                    first_img_disp = (first_img_disp - img_min) / (img_max - img_min)
                first_img_disp = first_img_disp * 255
                first_img_disp = first_img_disp.clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
            else:
                # If no normalization was applied, just use the first three channels
                first_img_disp = first_image[:3, :, :].permute(1, 2, 0).detach().cpu().numpy()
                # Scale to [0, 255] if needed
                if first_img_disp.max() <= 1.0:
                    first_img_disp = (first_img_disp * 255).astype(np.uint8)
                else:
                    first_img_disp = first_img_disp.clamp(0, 255).astype(np.uint8)

            # Get ground truth and prediction as numpy arrays
            gt_np = first_mask.detach().cpu().numpy().astype(np.int32)
            pred_np = first_prediction.detach().cpu().numpy().astype(np.int32)

            # Apply colormap to masks for better visualization
            colormap = cm.get_cmap('tab20', configs["num_classes"])
            
            # Normalize masks to [0, 1] for colormap
            if configs["num_classes"] > 1:
                gt_normalized = gt_np.astype(np.float32) / (configs["num_classes"] - 1)
                pred_normalized = pred_np.astype(np.float32) / (configs["num_classes"] - 1)
            else:
                gt_normalized = gt_np.astype(np.float32)
                pred_normalized = pred_np.astype(np.float32)
            
            # Apply colormap
            gt_colored = colormap(gt_normalized)[:, :, :3]  # Get RGB, ignore alpha
            pred_colored = colormap(pred_normalized)[:, :, :3]
            gt_colored = (gt_colored * 255).astype(np.uint8)
            pred_colored = (pred_colored * 255).astype(np.uint8)

            # Create side-by-side visualization using PIL (avoids matplotlib rendering issues)
            h, w = first_img_disp.shape[:2]
            combined_width = w * 3
            combined_image = np.zeros((h, combined_width, 3), dtype=np.uint8)
            
            # Place images side by side
            combined_image[:, 0:w, :] = first_img_disp
            combined_image[:, w:2*w, :] = gt_colored
            combined_image[:, 2*w:3*w, :] = pred_colored
            
            # Convert to PIL Image for saving/logging
            combined_pil = Image.fromarray(combined_image)
            
            # Save figure if checkpoint path is available
            if "checkpoint_path" in configs and configs["checkpoint_path"]:
                os.makedirs(configs["checkpoint_path"], exist_ok=True)
                save_path = os.path.join(configs["checkpoint_path"], f"{phase}_visualization_epoch_{epoch}.png")
                combined_pil.save(save_path)
                print(f"Saved visualization to: {save_path}")
            
            # Log to wandb if enabled
            if configs.get("wandb", False) and wandb.run:
                # Log the combined side-by-side image
                wandb.log({f"{phase}_visualization_combined": wandb.Image(combined_pil)})
                
                # Also log individual images for better wandb interaction
                wandb.log({
                    f"{phase}_raw_image": wandb.Image(first_img_disp),
                    f"{phase}_ground_truth": wandb.Image(gt_colored),
                    f"{phase}_prediction": wandb.Image(pred_colored),
                })
                print(f"✅ Logged visualization images to wandb: {phase}_visualization_combined, {phase}_raw_image, {phase}_ground_truth, {phase}_prediction")
                
        except Exception as e:
            # If visualization fails, just skip it and continue
            print(f"Warning: Could not generate visualization due to: {e}")
            print("Continuing without visualization...")

    # ---------------------------

    return total_loss