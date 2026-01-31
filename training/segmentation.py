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


def make_combined_visualization(image_tensor, mask_tensor, pred_tensor, configs):
    """Build side-by-side PIL image (raw | ground truth | prediction) for one sample.
    image_tensor: [C, H, W], mask_tensor: [H, W], pred_tensor: [H, W]. Returns PIL Image or None on error.
    """
    try:
        img = image_tensor
        if len(img.shape) == 3 and img.shape[0] > 100:
            img = img.permute(2, 0, 1)
        if configs["normalization"] == "standard":
            denorm_mean = torch.tensor(configs["mean"])[:3] if len(configs["mean"]) >= 3 else torch.tensor(configs["mean"])
            denorm_std = torch.tensor(configs["std"])[:3] if len(configs["std"]) >= 3 else torch.tensor(configs["std"])
            img_to_denorm = img[:3, :, :].unsqueeze(0)
            denormalizer = kornia.enhance.Denormalize(denorm_mean, denorm_std)
            img_disp = denormalizer(img_to_denorm).squeeze(0).permute(1, 2, 0)
            img_min, img_max = img_disp.min(), img_disp.max()
            if img_max > img_min:
                img_disp = (img_disp - img_min) / (img_max - img_min)
            img_disp = (img_disp * 255).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
        else:
            img_disp = img[:3, :, :].permute(1, 2, 0).detach().cpu().numpy()
            img_disp = (img_disp * 255).astype(np.uint8) if img_disp.max() <= 1.0 else np.clip(img_disp, 0, 255).astype(np.uint8)
        gt_np = mask_tensor.detach().cpu().numpy().astype(np.int32)
        pred_np = pred_tensor.detach().cpu().numpy().astype(np.int32)
        num_classes = configs["num_classes"]
        colormap = cm.get_cmap('tab20', num_classes)
        gt_norm = gt_np.astype(np.float32) / (num_classes - 1) if num_classes > 1 else gt_np.astype(np.float32)
        pred_norm = pred_np.astype(np.float32) / (num_classes - 1) if num_classes > 1 else pred_np.astype(np.float32)
        gt_colored = (colormap(gt_norm)[:, :, :3] * 255).astype(np.uint8)
        pred_colored = (colormap(pred_norm)[:, :, :3] * 255).astype(np.uint8)
        h, w = img_disp.shape[:2]
        combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
        combined[:, 0:w, :] = img_disp
        combined[:, w:2*w, :] = gt_colored
        combined[:, 2*w:3*w, :] = pred_colored
        return Image.fromarray(combined)
    except Exception:
        return None


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
                elif configs['architecture']=='maskedresnet_unet':
                    out, mask = model(image, label=label, keys=keys)
                elif configs['architecture']=='unet':   
                    out = model(image)
                else:
                    out = model(image, label=label)
            else:
                image = image[:, desired_indices, :, :]

                out = model((image, spectral_keys))

            label = label.long()
            if configs['architecture']=='rvitupernet' or configs['architecture']=='maskedresnet_unet':
                # Convert mask to float to maintain computation graph
                mask_float = mask.float()
                if mask_float.dim() == 2:
                    mask_float = mask_float.unsqueeze(0)
                if mask_float.dim() == 3 and out.shape[-2:] != mask_float.shape[-2:]:
                    mask_float = F.interpolate(
                        mask_float.unsqueeze(1).float(),
                        size=out.shape[-2:],
                        mode='nearest',
                    ).squeeze(1)
                # Combined CE+Dice (and weighted variant) support optional mask for CE
                if configs.get("loss") in ("ce_dice", "ce_dice_weighted"):
                    loss = criterion(out, label, mask=mask_float)
                else:
                    loss = F.cross_entropy(out, label, reduction='none')
                    if mask_float.dim() == 2:
                        mask_float = mask_float.expand_as(loss)
                    elif mask_float.shape != loss.shape:
                        mask_float = F.interpolate(
                            mask_float.unsqueeze(1).float(),
                            size=(loss.shape[1], loss.shape[2]),
                            mode='nearest',
                        ).squeeze(1)
                    mask_sum = mask_float.sum().clamp(min=1.0)
                    loss = (loss * mask_float).sum() / mask_sum
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
    import sys
    sys.stdout.flush()
    sys.stderr.flush()

    print("Initializing metrics...", flush=True)
    metrics = utils.initialize_metrics(configs)
    print("Creating dataloaders...", flush=True)
    # Create dataloaders before criterion so we can compute class weights for ce_dice_weighted
    train_loader, val_loader, _ = utils.create_dataloaders(configs)
    print("Dataloaders created.", flush=True)

    # Class weighting for macro Jaccard: compute from train set (log-inverse), then create loss
    if configs.get("loss") == "ce_dice_weighted":
        print("Computing class weights from training set (set class_weight_max_batches to limit, e.g. 100)...")
        class_weights = utils.compute_class_weights_log_inverse(
            train_loader,
            num_classes=configs["num_classes"],
            device=configs["device"],
            epsilon=configs.get("class_weight_epsilon", 1.02),
            max_weight=configs.get("class_weight_max", 10.0),
            max_batches=configs.get("class_weight_max_batches"),  # None = full pass
        )
        configs["class_weights"] = class_weights
        w = class_weights.cpu().numpy()
        print("[class_weights] log-inverse (epsilon={}): min={:.3f}, max={:.3f}, mean={:.3f}".format(
            configs.get("class_weight_epsilon", 1.02), float(w.min()), float(w.max()), float(w.mean())
        ))
        print("[class_weights] per-class:", [round(float(x), 3) for x in w])

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

    #train_loader, val_loader, _ = utils.create_dataloaders(configs)
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

        # Load model from checkpoint with error handling
        try:
            # Prefer explicit eval_checkpoint if provided; otherwise fall back to checkpoint_path/best_model.pt
            if "eval_checkpoint" in configs and configs["eval_checkpoint"] is not None:
                ckpt_path = configs["eval_checkpoint"]
                print("=" * 20)
                print("Evaluating segmentation for phase: ", phase, " with checkpoint: ")
                print(ckpt_path)
                print("=" * 20)
            else:
                ckpt_path = os.path.join(configs["checkpoint_path"], "best_model.pt")
                print("=" * 20)
                print("Evaluating segmentation for phase: ", phase, " with checkpoint: ")
                print(ckpt_path)
                print("=" * 20)

            # Try to load state_dict first (preferred for inference with RViT wrapper)
            ckpt_dir = os.path.dirname(ckpt_path)
            state_dict_path = os.path.join(ckpt_dir, "best_model_state_dict.pt")
            
            if os.path.isfile(state_dict_path):
                print(f"Loading model from state_dict: {state_dict_path}")
                model = model_utils.create_model(configs)
                state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"Warning: Missing keys when loading state_dict: {missing[:5]}...")
                if unexpected:
                    print(f"Warning: Unexpected keys when loading state_dict: {unexpected[:5]}...")
            else:
                # Fallback: load full model object
                print(f"Loading full model object from checkpoint: {ckpt_path}")
                model = torch.load(
                    ckpt_path,
                    map_location=configs["device"],
                    weights_only=False,
                )
                
                # If loaded model is not the expected type (e.g., not RViT/RViTUPerNet when rvit=True),
                # try to extract state_dict and load into a newly created model
                if configs.get("rvit", False) or configs.get("architecture") == "rvitupernet":
                    from model_zoo.RViT import RViT
                    from model_zoo.RViTUPerNet import RViTUPerNet
                    
                    # Check if loaded model is NOT an RViT/RViTUPerNet instance
                    is_rvit = isinstance(model, RViT)
                    is_rvitupernet = isinstance(model, RViTUPerNet)
                    
                    if not is_rvit and not is_rvitupernet:
                        print(f"[WARNING] Loaded model type ({type(model)}) is not RViT/RViTUPerNet. "
                              f"Extracting state_dict and loading into new RViT wrapper...")
                        # Extract state_dict from loaded model
                        if hasattr(model, "state_dict"):
                            old_state_dict = model.state_dict()
                            # Create new model with RViT wrapper
                            model = model_utils.create_model(configs)
                            missing, unexpected = model.load_state_dict(old_state_dict, strict=False)
                            if missing:
                                print(f"Warning: Missing keys when loading state_dict: {missing[:5]}...")
                            if unexpected:
                                print(f"Warning: Unexpected keys when loading state_dict: {unexpected[:5]}...")

            # --- Inference-time masking wiring for RViT-based models ---
            # Mark model as being in testing (inference) phase.
            # RViT checks this flag plus its internal apply_mask_in_infer flag.
            setattr(model, "testing", True)
            # Ensure inference-time masking flag matches current configs, even for older checkpoints
            # that were saved before the attribute existed.
            setattr(model, "apply_mask_in_infer", configs.get("apply_mask_in_infer", False))
            
            # For architectures that wrap an RViT backbone (e.g., RViTUPerNet),
            # also propagate the testing flag to the encoder/backbone if present.
            if hasattr(model, "encoder"):  # RViTUPerNet uses encoder
                setattr(model.encoder, "testing", True)
                setattr(model.encoder, "apply_mask_in_infer", configs.get("apply_mask_in_infer", False))
            elif hasattr(model, "backbone"):  # Some architectures use backbone
                setattr(model.backbone, "testing", True)
                setattr(model.backbone, "apply_mask_in_infer", configs.get("apply_mask_in_infer", False))

            # Debug: print model type and key masking-related attributes
            print(f"[DEBUG] Loaded model type: {type(model)}")
            if hasattr(model, "encoder"):
                print(f"[DEBUG] Model has encoder (RViTUPerNet): {type(model.encoder)}")
                if hasattr(model.encoder, "implementation"):
                    print(f"[DEBUG] RViT encoder implementation: {getattr(model.encoder, 'implementation')}")
                if hasattr(model.encoder, "retain_percent"):
                    print(f"[DEBUG] RViT encoder retain_percent: {getattr(model.encoder, 'retain_percent')}")
            elif hasattr(model, "implementation"):
                print(f"[DEBUG] RViT implementation: {getattr(model, 'implementation')}")
            if hasattr(model, "retain_percent"):
                print(f"[DEBUG] RViT retain_percent: {getattr(model, 'retain_percent')}")
            print(f"[DEBUG] apply_mask_in_infer: {getattr(model, 'apply_mask_in_infer', None)}")
            print(f"[DEBUG] testing flag on model: {getattr(model, 'testing', None)}")
        except Exception as e:
            print(f"Error loading model from checkpoint: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: create a fresh model so evaluation can still run
            model = model_utils.create_model(configs)
            setattr(model, "testing", True)
            setattr(model, "apply_mask_in_infer", configs.get("apply_mask_in_infer", False))
            if hasattr(model, "encoder"):
                setattr(model.encoder, "testing", True)
                setattr(model.encoder, "apply_mask_in_infer", configs.get("apply_mask_in_infer", False))

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

    # Collect up to num_vis_images samples for combined (raw|GT|pred) visualization
    num_vis_images = configs.get("num_vis_images", 15)
    vis_samples = []  # list of (image, mask, prediction) tensors
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
                    elif configs['architecture'] == 'maskedresnet_unet':
                        out, _ = model(image, label=label, keys=keys)  # Ignore mask in test
                    elif configs['architecture'] == 'unet':
                        out = model(image)
                    else:
                        out = model(image, label=label)
                else:
                    image = image[:, desired_indices, :, :]
                    out = model((image, spectral_keys))

                # if idx == 0:
                #     predictions = out.argmax(1)
                predictions = out.argmax(1)
                # Collect samples for combined visualization (up to num_vis_images)
                for i in range(image.shape[0]):
                    if len(vis_samples) >= num_vis_images:
                        break
                    vis_samples.append((
                        image.detach().cpu()[i],
                        label.detach().cpu()[i],
                        predictions.detach().cpu()[i],
                    ))
                if first_image is None and image.shape[0] > 0:
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
        elif getattr(metric, "flair_12", False):
            # FLAIR paper metric: mIoU on first 12 classes only (indices 0–11, excluding "other")
            per_class = metric.compute()
            if per_class is not None and per_class.numel() >= 12:
                first_12 = per_class[:12]
                # Use nanmean so classes with no GT in this split don't make the metric NaN
                miou_12 = torch.nanmean(first_12.to(torch.float32)).item()
                if not (miou_12 == miou_12):  # NaN check
                    miou_12 = 0.0
                log_dict[phase + " macro JaccardIndex_12 (FLAIR paper)"] = miou_12
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
    # if configs.get("log_images", False) and first_image is not None and first_mask is not None and first_prediction is not None:
       # Visualization: side-by-side (raw | ground truth | prediction) for up to num_vis_images samples
    if configs.get("log_images", False) and vis_samples:
        try:
            # # Prepare raw image for visualization (first 3 channels, denormalized if needed)
            # if configs["normalization"] == "standard":
            #     # Extract only the first three values from the mean and std
            #     denorm_mean = torch.tensor(configs["mean"])[:3] if len(configs["mean"]) >= 3 else torch.tensor(configs["mean"])
            #     denorm_std = torch.tensor(configs["std"])[:3] if len(configs["std"]) >= 3 else torch.tensor(configs["std"])
            #     # Extract the first three channels from the image for visualization
            #     img_to_denorm = first_image[:3, :, :].unsqueeze(0)  # shape: [1, 3, H, W]
            #     denormalizer = kornia.enhance.Denormalize(denorm_mean, denorm_std)
            #     first_img_denorm = denormalizer(img_to_denorm).squeeze(0)
            #     # Permute to H, W, C for display and scale to [0, 255]
            #     first_img_disp = first_img_denorm.permute(1, 2, 0)
            #     # Normalize to [0, 1] then scale to [0, 255]
            #     img_min = first_img_disp.min()
            #     img_max = first_img_disp.max()
            #     if img_max > img_min:
            #         first_img_disp = (first_img_disp - img_min) / (img_max - img_min)
            #     first_img_disp = first_img_disp * 255
            #     first_img_disp = first_img_disp.clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
            # else:
            #     # If no normalization was applied, just use the first three channels
            #     first_img_disp = first_image[:3, :, :].permute(1, 2, 0).detach().cpu().numpy()
            #     # Scale to [0, 255] if needed
            #     if first_img_disp.max() <= 1.0:
            #         first_img_disp = (first_img_disp * 255).astype(np.uint8)
            #     else:
            #         first_img_disp = first_img_disp.clamp(0, 255).astype(np.uint8)

            # # Get ground truth and prediction as numpy arrays
            # gt_np = first_mask.detach().cpu().numpy().astype(np.int32)
            # pred_np = first_prediction.detach().cpu().numpy().astype(np.int32)

            # # Apply colormap to masks for better visualization
            # colormap = cm.get_cmap('tab20', configs["num_classes"])
            #
            # # Normalize masks to [0, 1] for colormap
            # if configs["num_classes"] > 1:
            #     gt_normalized = gt_np.astype(np.float32) / (configs["num_classes"] - 1)
            #     pred_normalized = pred_np.astype(np.float32) / (configs["num_classes"] - 1)
            # else:
            #     gt_normalized = gt_np.astype(np.float32)
            #     pred_normalized = pred_np.astype(np.float32)
            #
            # # Apply colormap
            # gt_colored = colormap(gt_normalized)[:, :, :3]  # Get RGB, ignore alpha
            # pred_colored = colormap(pred_normalized)[:, :, :3]
            # gt_colored = (gt_colored * 255).astype(np.uint8)
            # pred_colored = (pred_colored * 255).astype(np.uint8)

            # # Create side-by-side visualization using PIL (avoids matplotlib rendering issues)
            # h, w = first_img_disp.shape[:2]
            # combined_width = w * 3
            # combined_image = np.zeros((h, combined_width, 3), dtype=np.uint8)
            #
            # # Place images side by side
            # combined_image[:, 0:w, :] = first_img_disp
            # combined_image[:, w:2*w, :] = gt_colored
            # combined_image[:, 2*w:3*w, :] = pred_colored
            #
            # # Convert to PIL Image for saving/logging
            # combined_pil = Image.fromarray(combined_image)
            #
            # # Save figure if checkpoint path is available
            # if "checkpoint_path" in configs and configs["checkpoint_path"]:
            #     os.makedirs(configs["checkpoint_path"], exist_ok=True)
            #     save_path = os.path.join(configs["checkpoint_path"], f"{phase}_visualization_epoch_{epoch}.png")
            #     combined_pil.save(save_path)
            #     print(f"Saved visualization to: {save_path}")
            #
            # # Log to wandb if enabled
            # if configs.get("wandb", False) and wandb.run:
            #     # Log the combined side-by-side image
            #     wandb.log({f"{phase}_visualization_combined": wandb.Image(combined_pil)})
            #
            #     # Also log individual images for better wandb interaction
            #     # wandb.log({
            #     #     f"{phase}_raw_image": wandb.Image(first_img_disp),
            #     #     f"{phase}_ground_truth": wandb.Image(gt_colored),
            #     #     f"{phase}_prediction": wandb.Image(pred_colored),
            #     # })
            #     print(f"✅ Logged visualization images to wandb: {phase}_visualization_combined, {phase}_raw_image, {phase}_ground_truth, {phase}_prediction")
            # Current logic: up to num_vis_images samples, each via make_combined_visualization(); log combined raw|GT|pred to wandb.
            log_dict_vis = {}
            for i, (img_t, mask_t, pred_t) in enumerate(vis_samples):
                combined_pil = make_combined_visualization(img_t, mask_t, pred_t, configs)
                if combined_pil is None:
                    continue
                if "checkpoint_path" in configs and configs["checkpoint_path"]:
                    os.makedirs(configs["checkpoint_path"], exist_ok=True)
                    save_path = os.path.join(configs["checkpoint_path"], f"{phase}_visualization_epoch_{epoch}_sample_{i}.png")
                    combined_pil.save(save_path)
                log_dict_vis[f"{phase}_visualization_combined_{i}"] = wandb.Image(combined_pil, caption=f"Sample {i}: raw | GT | pred")
            if log_dict_vis and configs.get("wandb", False) and wandb.run:
                wandb.log(log_dict_vis)
                print(f"✅ Logged {len(log_dict_vis)} combined visualizations to wandb: {phase}_visualization_combined_0 .. _{len(log_dict_vis) - 1}")
        except Exception as e:
            print(f"Warning: Could not generate visualization due to: {e}")
            print("Continuing without visualization...")

    # ---------------------------

    return total_loss