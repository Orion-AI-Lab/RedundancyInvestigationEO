import math
import sys

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from torchsummary import summary
import einops
#from timm.models.layers import PatchEmbed
from model_zoo import RViT, RViTUPerNet ,upernet 
import yaml
from pathlib import Path
import kornia
from functools import partial


def initialize_rgb_pretrained_weights(model, backbone, image_size, num_classes, num_channels, rgb_channel_indices=None):
    """
    Override timm's automatic weight conversion with a better strategy:
    - RGB channels: Use ImageNet pretrained weights directly
    - Additional channels: Randomly initialize
    
    
    Args:
        model: The model (already created by timm with pretrained=True, in_chans=num_channels)
        backbone: The backbone name (e.g., 'vit_base_patch16_224', 'resnet50')
        image_size: Image size
        num_classes: Number of classes
        num_channels: Total number of channels
        rgb_channel_indices: List of 3 indices for RGB channels [R, G, B]. 
                           Default is [0, 1, 2] (for reordered data).
                           For BigEarthNet, use [5, 4, 3] (B04=Red, B03=Green, B02=Blue; indices per modality order VV,VH,B01,B02,B03,B04,...).
                           For FLAIR, use [0, 1, 2] (aerial-r=Red, aerial-g=Green, aerial-b=Blue).
    
    Returns:
        model: Model with optimized weight initialization
    """
    if num_channels <= 3:
        # No need to override for 3 channels or less
        return model
    
    # Default RGB channel indices (for reordered data where RGB is first)
    if rgb_channel_indices is None:
        rgb_channel_indices = [0, 1, 2]
    
    if len(rgb_channel_indices) != 3:
        raise ValueError("rgb_channel_indices must contain exactly 3 indices [R, G, B]")
    
    r_idx, g_idx, b_idx = rgb_channel_indices
    
    print(f"=" * 60)
    print(f"Overriding timm's weight conversion with optimized RGB initialization")
    print(f"  - RGB channels ({r_idx}, {g_idx}, {b_idx}): ImageNet pretrained weights (direct copy)")
    print(f"  - Additional channels: Random initialization")
    print(f"=" * 60)
    
    try:
        # Create a pretrained model with 3 channels to get original RGB weights
        # Only pass img_size for Vision Transformers (ResNet doesn't support it)
        create_kwargs = {
            "pretrained": True,
            "in_chans": 3,
            "num_classes": num_classes,
        }
        # Only add img_size parameters for ViT models (ResNet doesn't support them)
        if "vit" in backbone.lower() or "swin" in backbone.lower() or "deit" in backbone.lower():
            create_kwargs["img_size"] = image_size
            create_kwargs["dynamic_img_pad"] = True
            create_kwargs["dynamic_img_size"] = True
        
        pretrained_model = timm.create_model(backbone.lower(), **create_kwargs)
        
        # Get the model to modify
        if hasattr(model, 'model'):  # For RViT wrapper
            model_to_modify = model.model
        elif hasattr(model, 'encoder'):  # For smp models (UNet, DeepLab, etc.)
            model_to_modify = model.encoder
        else:
            model_to_modify = model
        
        # Find the first layer (patch embedding for ViT, conv1 for ResNet, etc.)
        first_layer = None
        pretrained_first_layer = None
        first_layer_name = None
            
        # Check for Vision Transformer patch embedding
        if hasattr(model_to_modify, 'patch_embed') and hasattr(model_to_modify.patch_embed, 'proj'):
            first_layer = model_to_modify.patch_embed.proj
            first_layer_name = 'patch_embed.proj'
            pretrained_first_layer = pretrained_model.patch_embed.proj
        # Check for ResNet conv1 (standard timm structure)
        elif hasattr(model_to_modify, 'conv1'):
            first_layer = model_to_modify.conv1
            first_layer_name = 'conv1'
            pretrained_first_layer = pretrained_model.conv1
        # Check for ResNet layer0 (smp structure for some encoders)
        elif hasattr(model_to_modify, 'layer0') and hasattr(model_to_modify.layer0, 'conv1'):
            first_layer = model_to_modify.layer0.conv1
            first_layer_name = 'layer0.conv1'
            # For smp, we need to check if pretrained_model has layer0 or just conv1
            if hasattr(pretrained_model, 'layer0') and hasattr(pretrained_model.layer0, 'conv1'):
                pretrained_first_layer = pretrained_model.layer0.conv1
            elif hasattr(pretrained_model, 'conv1'):
                pretrained_first_layer = pretrained_model.conv1
            else:
                pretrained_first_layer = None
        # Check for ResNet stem (newer architectures)
        elif hasattr(model_to_modify, 'stem') and len(model_to_modify.stem) > 0:
            first_layer = model_to_modify.stem[0]
            first_layer_name = 'stem.0'
            pretrained_first_layer = pretrained_model.stem[0]
        # Check for ConvNeXt stem
        elif hasattr(model_to_modify, 'stem') and hasattr(model_to_modify.stem, 'conv'):
            first_layer = model_to_modify.stem.conv
            first_layer_name = 'stem.conv'
            pretrained_first_layer = pretrained_model.stem.conv
        else:
            print("Warning: Could not find first layer. Keeping timm's default conversion.")
            print(f"Available attributes: {[attr for attr in dir(model_to_modify) if not attr.startswith('_')]}")
            del pretrained_model
            return model
        
        if first_layer is None or pretrained_first_layer is None:
            print("Warning: Could not locate first layer. Keeping timm's default conversion.")
            del pretrained_model
            return model
        
        # Get original RGB weights (3 channels) from pretrained model
        pretrained_weight = pretrained_first_layer.weight.data.clone()
        pretrained_bias = pretrained_first_layer.bias.data.clone() if pretrained_first_layer.bias is not None else None
        
        # Get current layer shape (already converted by timm to num_channels)
        out_channels, current_in_channels, kernel_h, kernel_w = first_layer.weight.shape
        
        # Verify shapes match for output channels and kernel size
        if pretrained_weight.shape[0] != out_channels or pretrained_weight.shape[2:] != (kernel_h, kernel_w):
            print(f"Warning: Shape mismatch in first layer ({pretrained_weight.shape} vs {first_layer.weight.shape})")
            print("Keeping timm's default conversion.")
            del pretrained_model
            return model
        
        # Override timm's conversion: use RGB weights directly, randomly initialize rest
        # Create new weight tensor from scratch (more efficient than cloning timm's converted weights)
        new_weight = torch.zeros_like(first_layer.weight.data)
        
        # Copy RGB channels with original ImageNet pretrained weights
        # pretrained_weight shape: [out_channels, 3, kernel_h, kernel_w]
        # We need to map: ImageNet R -> channel r_idx, ImageNet G -> channel g_idx, ImageNet B -> channel b_idx
        new_weight[:, r_idx, :, :] = pretrained_weight[:, 0, :, :]  # Red
        new_weight[:, g_idx, :, :] = pretrained_weight[:, 1, :, :]  # Green
        new_weight[:, b_idx, :, :] = pretrained_weight[:, 2, :, :]  # Blue
        
        # Debug: Verify the copy worked
        red_match = torch.allclose(new_weight[:, r_idx, :, :], pretrained_weight[:, 0, :, :], atol=1e-6)
        green_match = torch.allclose(new_weight[:, g_idx, :, :], pretrained_weight[:, 1, :, :], atol=1e-6)
        blue_match = torch.allclose(new_weight[:, b_idx, :, :], pretrained_weight[:, 2, :, :], atol=1e-6)
        
        if not (red_match and green_match and blue_match):
            print(f"  ⚠️  Warning: RGB weight copy verification failed!")
            print(f"     Red match: {red_match}, Green match: {green_match}, Blue match: {blue_match}")
        
        # Randomly initialize remaining channels using Kaiming initialization
        remaining_channels = [i for i in range(num_channels) if i not in rgb_channel_indices]
        if remaining_channels:
            for ch_idx in remaining_channels:
                nn.init.kaiming_normal_(new_weight[:, ch_idx:ch_idx+1, :, :], mode='fan_out', nonlinearity='relu')
        
        # Set the optimized weights
        first_layer.weight.data = new_weight
        
        # Handle bias if present (bias doesn't depend on input channels, so keep timm's)
        if pretrained_bias is not None and first_layer.bias is not None:
            first_layer.bias.data = pretrained_bias.clone()
        
        print(f"  ✓ Optimized first layer '{first_layer_name}'")
        print(f"    - RGB channels ({r_idx}, {g_idx}, {b_idx}): Original ImageNet pretrained weights")
        print(f"    - Additional channels: Random initialization (Kaiming)")
        print(f"  ✓ Override complete - timm's conversion has been replaced with optimal initialization")
        print(f"=" * 60)
        
        # Clean up pretrained model
        del pretrained_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"Error during RGB pretrained initialization: {e}")
        print("Keeping timm's default conversion.")
        import traceback
        traceback.print_exc()
    
    return model


def load_segmentation_encoder_into_classifier(classifier_model, checkpoint_path, strict=False):
    """
    Load encoder weights from a FLAIR (or other) segmentation checkpoint into an RViT classifier.
    Segmentation model (RViTUPerNet) stores the backbone as encoder; keys are "encoder.model.*".
    Classifier (RViT) expects "model.*". We map encoder.X -> X and load only matching keys.
    If the segmentation backbone was 3-channel (e.g. FLAIR RGB) and the classifier is multi-channel
    (e.g. BigEarthNet 14ch), we copy the 3-channel patch_embed.proj into the first 3 input channels.
    Pos_embed is interpolated when resolution differs (e.g. FLAIR 512px -> BigEarthNet 128px).
    """
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif not isinstance(ckpt, dict):
        ckpt = getattr(ckpt, "state_dict", lambda: ckpt)()
    # Support DDP-saved checkpoints: strip "module." so "module.encoder.*" -> "encoder.*"
    ckpt = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in ckpt.items()}
    model_sd = classifier_model.state_dict()
    prefix = "encoder."
    filtered = {}
    patch_embed_key = None
    patch_embed_val = None
    # Find classifier's pos_embed key (timm may use model._pos_embed; state_dict can show model.pos_embed)
    pos_embed_target_key = None
    for pk in ["model.pos_embed", "model._pos_embed"]:
        if pk in model_sd:
            pos_embed_target_key = pk
            break

    for k, v in ckpt.items():
        if not k.startswith(prefix):
            continue
        new_k = k[len(prefix) :]
        if new_k not in model_sd:
            # Still handle pos_embed: checkpoint may have model._pos_embed, classifier may have model.pos_embed
            if pos_embed_target_key and ("pos_embed" in new_k or "_pos_embed" in new_k) and v.dim() == 3:
                target = model_sd[pos_embed_target_key]
                if v.shape[2] == target.shape[2]:
                    if v.shape[1] != target.shape[1]:
                        num_extra = 1
                        orig_np = v.shape[1] - num_extra
                        new_np = target.shape[1] - num_extra
                        resized = interpolate_pos_embed_2d(
                            v.clone(), num_extra_tokens=num_extra,
                            orig_num_patches=orig_np, new_num_patches=new_np, mode="bicubic"
                        )
                        orig_side = int(orig_np ** 0.5)
                        print(f"  Resized pos_embed (2D bicubic): {v.shape[1]} -> {target.shape[1]} positions (checkpoint grid {orig_side}×{orig_side})")
                        filtered[pos_embed_target_key] = resized
                    else:
                        filtered[pos_embed_target_key] = v.clone()
            continue
        target = model_sd[new_k]
        if v.shape == target.shape:
            filtered[new_k] = v
        elif new_k == "model.patch_embed.proj.weight" and v.dim() == 4:
            # Segmentation: 3 or 5 ch (e.g. FLAIR RGB or RGB+NIR+DEM), classifier: more ch (e.g. BigEarthNet 14)
            if v.shape[0] == target.shape[0] and v.shape[2:] == target.shape[2:] and v.shape[1] >= 3 and target.shape[1] >= 3:
                patch_embed_key = new_k
                patch_embed_val = v  # use first 3 channels (RGB) when copying into target
            # else skip
        elif pos_embed_target_key and ("pos_embed" in new_k or "_pos_embed" in new_k) and v.dim() == 3 and target.dim() == 3:
            # Interpolate if resolution differs
            if v.shape[2] == target.shape[2] and v.shape[1] != target.shape[1]:
                num_extra = 1
                orig_np = v.shape[1] - num_extra
                new_np = target.shape[1] - num_extra
                resized = interpolate_pos_embed_2d(
                    v.clone(), num_extra_tokens=num_extra,
                    orig_num_patches=orig_np, new_num_patches=new_np, mode="bicubic"
                )
                orig_side = int(orig_np ** 0.5)
                print(f"  Resized pos_embed (2D bicubic): {v.shape[1]} -> {target.shape[1]} positions (checkpoint grid {orig_side}×{orig_side})")
                filtered[pos_embed_target_key] = resized
            elif v.shape == target.shape:
                filtered[new_k] = v
                if pos_embed_target_key != new_k:
                    filtered[pos_embed_target_key] = v.clone()
        # else skip (e.g. different num_classes head)
    if patch_embed_key and patch_embed_val is not None:
        target = model_sd[patch_embed_key].clone()
        n_copy = min(3, patch_embed_val.shape[1], target.shape[1])
        target[:, :n_copy, :, :] = patch_embed_val[:, :n_copy, :, :].clone()
        filtered[patch_embed_key] = target
        print(f"  Transferred patch_embed.proj: first {n_copy} ch from checkpoint -> first {n_copy} ch of classifier ({target.shape[1]} ch)")
    missing, unexpected = classifier_model.load_state_dict(filtered, strict=False)
    n_loaded = len(filtered)
    print(f"  Loaded {n_loaded} keys from segmentation checkpoint (encoder -> classifier backbone).")
    if missing:
        print(f"  Missing in checkpoint (kept random/init): {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"  Unexpected (ignored): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    return classifier_model


def interpolate_pos_embed_2d(pos_embed, num_extra_tokens, orig_num_patches, new_num_patches, mode="bicubic"):
    """
    Resize positional embeddings using 2D interpolation on the patch grid.

    The spatial part of pos_embed (excluding class token) is stored as a flat sequence
    (1, num_patches, C) but corresponds to a 2D grid of patches (e.g. 7×7 or 16×16).
    This function:
      1. Splits off the class token(s).
      2. Reshapes the spatial part to (1, orig_h, orig_w, C) then (1, C, orig_h, orig_w).
      3. Runs 2D bicubic interpolation to (new_h, new_w), so each new position is a
         blend of spatially nearby positions in the image (not arbitrary sequence neighbors).
      4. Reshapes back to (1, new_num_patches, C) and concatenates the class token.

    See docs/POS_EMBED_2D_INTERPOLATION.md for a longer explanation.
    Uses kornia.geometry.transform.resize for 2D resizing (same library as rest of project;
    supports bicubic and optional antialias for downscaling).
    """
    if orig_num_patches == new_num_patches:
        return pos_embed
    orig_h = int(orig_num_patches ** 0.5)
    orig_w = orig_num_patches // orig_h
    new_h = int(new_num_patches ** 0.5)
    new_w = new_num_patches // new_h
    if orig_h * orig_w != orig_num_patches or new_h * new_w != new_num_patches:
        # Non-square grid: fallback to 1D interpolation along the sequence
        perm = pos_embed.permute(0, 2, 1)
        perm_interp = torch.nn.functional.interpolate(
            perm, size=new_num_patches + num_extra_tokens, mode="linear", align_corners=False
        )
        return perm_interp.permute(0, 2, 1)
    extra = pos_embed[:, :num_extra_tokens, :]
    spatial = pos_embed[:, num_extra_tokens:, :]
    # (1, N, C) -> (1, H, W, C) -> (1, C, H, W); last two dims are H, W for resize
    spatial = spatial.reshape(1, orig_h, orig_w, -1).permute(0, 3, 1, 2)
    spatial = kornia.geometry.transform.resize(
        spatial, size=(new_h, new_w), interpolation=mode, align_corners=False
    )
    spatial = spatial.permute(0, 2, 3, 1).reshape(1, new_num_patches, -1)
    return torch.cat([extra, spatial], dim=1)


def load_classification_encoder_into_classifier(classifier_model, checkpoint_path, strict=False, source_rgb_indices=None):
    """
    Load encoder weights from a classification checkpoint (e.g. BigEarthNet-trained RViT) into
    a classifier created for a different dataset (e.g. MLRSNet). Handles channel and resolution
    adaptation so you can freeze the encoder and train only the head on the target dataset.

    - If checkpoint has more input channels (e.g. 14) than target (e.g. 3), copies the first 3
      (or source_rgb_indices) channels of patch_embed.proj into the target.
    - If pos_embed shape differs (e.g. 120px vs 256px), interpolates positional embeddings.
    """
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Classification checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif not isinstance(ckpt, dict):
        ckpt = getattr(ckpt, "state_dict", lambda: ckpt)()
    # If loaded model was FinetunerClassification, state_dict has "model.*" and "head.*"; we only want encoder
    model_sd = classifier_model.state_dict()
    filtered = {}
    patch_embed_key = "model.patch_embed.proj.weight"
    pos_embed_key = None
    for k in ["model.pos_embed", "model._pos_embed"]:
        if k in ckpt and k in model_sd:
            pos_embed_key = k
            break
    if pos_embed_key is None and "model.pos_embed" in ckpt:
        pos_embed_key = "model.pos_embed"
    if pos_embed_key is None:
        for k in model_sd:
            if "pos_embed" in k:
                pos_embed_key = k
                if pos_embed_key not in ckpt:
                    pos_embed_key = None
                break

    for k, v in ckpt.items():
        if k not in model_sd:
            continue
        target = model_sd[k]
        if v.shape == target.shape:
            filtered[k] = v
        elif k == patch_embed_key and v.dim() == 4:
            # Source has more (or same) in_channels; target has fewer (e.g. 3 for MLRSNet)
            if v.shape[0] == target.shape[0] and v.shape[2:] == target.shape[2:] and v.shape[1] >= target.shape[1]:
                if source_rgb_indices is not None:
                    sel = source_rgb_indices[: target.shape[1]]
                    filtered[k] = v[:, sel, :, :].clone()
                else:
                    filtered[k] = v[:, : target.shape[1], :, :].clone()
                print(f"  Transferred patch_embed.proj: {v.shape[1]} ch -> {target.shape[1]} ch (first channels)")
        elif pos_embed_key and k == pos_embed_key and v.dim() == 3 and target.dim() == 3:
            if v.shape[2] == target.shape[2] and v.shape[1] != target.shape[1]:
                # 2D bicubic interpolation on the patch grid (class token unchanged, spatial resized in 2D)
                num_extra = 1
                orig_np = v.shape[1] - num_extra
                new_np = target.shape[1] - num_extra
                filtered[k] = interpolate_pos_embed_2d(
                    v.clone(), num_extra_tokens=num_extra,
                    orig_num_patches=orig_np, new_num_patches=new_np, mode="bicubic"
                )
                # Infer source grid: 50 positions = 7×7 (image_size 112), 65 = 8×8 (128), etc.
                orig_side = int(orig_np ** 0.5)
                inferred_src_size = orig_side * 16 if orig_side * orig_side == orig_np else None
                hint = f" (checkpoint grid {orig_side}×{orig_side}" + (f", ~{inferred_src_size}×{inferred_src_size} with patch_size 16" if inferred_src_size else "") + ")"
                print(f"  Resized pos_embed (2D bicubic): {v.shape[1]} -> {target.shape[1]} positions{hint}")
            elif v.shape == target.shape:
                filtered[k] = v

    missing, unexpected = classifier_model.load_state_dict(filtered, strict=False)
    n_loaded = len(filtered)
    print(f"  Loaded {n_loaded} encoder keys from classification checkpoint.")
    if missing:
        print(f"  Missing in checkpoint (kept random/init): {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"  Unexpected (ignored): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    return classifier_model


def create_classifier(configs):
    if "resnet_star_evaluation" in configs and configs["resnet_star_evaluation"]:
        model = torch.load(configs["resnet_star_checkpoint"], map_location=torch.device("cpu"))
        model = model.base_model
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, configs["num_classes"])
        print("Loaded ResNet* model")
        return model
    img_size = configs.get("effective_image_size", configs["image_size"])
    if configs['backbone'].lower() == 'act_base':
        import model_zoo.ACTViT as ACTViT
        model = ACTViT.AgglomerativeClusteringVisionTransformer(
                patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, img_size=img_size,in_chans=configs['in_channels'],num_classes=configs['num_classes'],
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), configs=configs)
    elif "rvit" in configs and configs["rvit"]:
        # Segmentation-to-classification transfer only when explicitly enabled in config (seg_to_class + checkpoint path)
        transfer_ckpt = configs.get("segmentation_pretrained_checkpoint") or configs.get("transfer_from_segmentation_checkpoint")
        if configs.get("seg_to_class", False) and transfer_ckpt:
            # Transfer from segmentation (e.g. FLAIR): create model without ImageNet, then load encoder from checkpoint
            model = RViT.RViT(
                num_classes=configs['num_classes'],
                backbone=configs['backbone'],
                image_size=img_size,
                num_channels=configs["in_channels"],
                pretrained=False,
                mask_threshold=configs["mask_threshold"],
                mask_in_token_space=configs['mask_in_token_space'],
                implementation=configs['implementation'],
                pad_value=configs['pad_value'],
                retain_percent=configs['retain_percent'],
                attention_mask=configs['attention_mask'],
                apply_mask_in_infer=configs.get("apply_mask_in_infer", False),
            )
            print("Transferring encoder weights from segmentation checkpoint (e.g. FLAIR) into classifier.")
            load_segmentation_encoder_into_classifier(model, transfer_ckpt, strict=False)
        else:
            # Original behaviour: create RViT with pretrained (timm), then optional RGB override for multi-channel
            model = RViT.RViT(
                num_classes=configs['num_classes'],
                backbone=configs['backbone'],
                image_size=img_size,
                num_channels=configs["in_channels"],
                pretrained=configs['pretrained'],
                mask_threshold=configs["mask_threshold"],
                mask_in_token_space=configs['mask_in_token_space'],
                implementation=configs['implementation'],
                pad_value=configs['pad_value'],
                retain_percent=configs['retain_percent'],
                attention_mask=configs['attention_mask'],
                apply_mask_in_infer=configs.get("apply_mask_in_infer", False),
            )
            if configs['pretrained'] and configs["in_channels"] > 3:
                # Determine RGB channel indices based on dataset
                rgb_indices = None
                if configs.get('dataset') == 'bigearthnet':
                    # BigEarthNet modality index: B04=5 (Red), B03=4 (Green), B02=3 (Blue); order VV,VH,B01,B02,B03,B04,...
                    if configs.get('rgb_channels_reordered', False):
                        rgb_indices = [0, 1, 2]
                    else:
                        rgb_indices = [5, 4, 3]
                elif configs.get('dataset') == 'flair' or configs.get('dataset') == 'flair2':
                    rgb_indices = [0, 1, 2]
                model = initialize_rgb_pretrained_weights(
                    model, configs['backbone'], img_size,
                    configs['num_classes'], configs["in_channels"], rgb_channel_indices=rgb_indices
                )
    else:
        # Create model with timm (it will handle pretrained conversion automatically)
        model = timm.create_model(
            configs["backbone"].lower(),
            pretrained=configs["pretrained"],
            in_chans=configs["in_channels"],
            num_classes=configs["num_classes"],
            img_size=img_size, dynamic_img_pad=True, dynamic_img_size=True,
            #patch_size=configs["patch_size"]
        )
        # Override timm's weight conversion if pretrained and multi-channel
        if configs['pretrained'] and configs["in_channels"] > 3:
            # Determine RGB channel indices based on dataset
            rgb_indices = None
            if configs.get('dataset') == 'bigearthnet':
                # BigEarthNet modality index: RGB at indices [5, 4, 3] (B04=Red, B03=Green, B02=Blue)
                if configs.get('rgb_channels_reordered', False):
                    rgb_indices = [0, 1, 2]  # Reordered: RGB first
                else:
                    rgb_indices = [5, 4, 3]  # Modality order: VV,VH,B01,B02,B03,B04,... → B04,B03,B02
            elif configs.get('dataset') == 'flair' or configs.get('dataset') == 'flair2':
                # FLAIR modality index: aerial-r=0 (Red), aerial-g=1 (Green), aerial-b=2 (Blue)
                rgb_indices = [0, 1, 2]  # RGB at first 3 channels
            
            model = initialize_rgb_pretrained_weights(
                model, configs['backbone'], img_size, 
                configs['num_classes'], configs["in_channels"], rgb_channel_indices=rgb_indices
            )
        return model
    return model


def create_segmentor(configs):
    img_size = configs.get("effective_image_size", configs["image_size"])
    if configs["architecture"].lower() == "unet":
        model = smp.Unet(
            encoder_name=configs["backbone"],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet"
            if configs["pretrained"]
            else None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=configs["in_channels"],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=configs["num_classes"],  # model output channels (number of classes in your dataset)
        )
        # Override encoder's first layer with optimized RGB initialization if pretrained and multi-channel
        if configs['pretrained'] and configs["in_channels"] > 3:
            # Determine RGB channel indices based on dataset
            rgb_indices = None
            if configs.get('dataset') == 'bigearthnet':
                if configs.get('rgb_channels_reordered', False):
                    rgb_indices = [0, 1, 2]  # Reordered: RGB first
                else:
                    rgb_indices = [5, 4, 3]  # Modality order: VV,VH,B01,B02,B03,B04,... → B04,B03,B02
            elif configs.get('dataset') == 'flair' or configs.get('dataset') == 'flair2':
                # FLAIR modality index: aerial-r=0 (Red), aerial-g=1 (Green), aerial-b=2 (Blue)
                rgb_indices = [0, 1, 2]  # RGB at first 3 channels
            
            # Apply RGB initialization to the encoder (smp uses timm encoders)
            model.encoder = initialize_rgb_pretrained_weights(
                model.encoder, configs['backbone'], img_size,
                configs['num_classes'], configs["in_channels"], rgb_channel_indices=rgb_indices
            )
    elif configs["architecture"].lower() == "rvitupernet":
        # Create backbone with pretrained (timm will handle conversion)
        backbone = RViT.RViT(
            num_classes=configs['num_classes'],
            backbone=configs['backbone'],
            image_size=img_size,
            num_channels=configs["in_channels"],
            pretrained=configs['pretrained'],
            mask_threshold=configs["mask_threshold"],
            mask_in_token_space=configs['mask_in_token_space'],
            implementation=configs['implementation'],
            pad_value=configs['pad_value'],
            retain_percent=configs['retain_percent'],
            attention_mask=configs['attention_mask'],
            hierarchical_features=True,
            apply_mask_in_infer=configs.get("apply_mask_in_infer", False),
        )
        # Override timm's weight conversion if pretrained and multi-channel
        if configs['pretrained'] and configs["in_channels"] > 3:
            # Determine RGB channel indices based on dataset
            rgb_indices = None
            if configs.get('dataset') == 'bigearthnet':
                # BigEarthNet modality index: RGB at indices [5, 4, 3] (B04=Red, B03=Green, B02=Blue)
                if configs.get('rgb_channels_reordered', False):
                    rgb_indices = [0, 1, 2]  # Reordered: RGB first
                else:
                    rgb_indices = [5, 4, 3]  # Modality order: VV,VH,B01,B02,B03,B04,... → B04,B03,B02
            elif configs.get('dataset') == 'flair' or configs.get('dataset') == 'flair2':
                # FLAIR modality index: aerial-r=0 (Red), aerial-g=1 (Green), aerial-b=2 (Blue)
                rgb_indices = [0, 1, 2]  # RGB at first 3 channels
            elif configs.get('dataset') == 'waititu':
                # Waititu: first 3 bands are RGB; 4th is alpha or NIR — use RGB for pretrained, init 4th
                rgb_indices = [0, 1, 2]
            elif configs.get('dataset') == 'marida':
                # MARIDA: 11 Sentinel-2 bands (ACOLITE Rayleigh); first 3 at 10m are B02, B03, B04 (Blue, Green, Red)
                rgb_indices = [0, 1, 2]

            backbone = initialize_rgb_pretrained_weights(
                backbone, configs['backbone'], img_size,
                configs['num_classes'], configs["in_channels"], rgb_channel_indices=rgb_indices
            )
        model = RViTUPerNet.RViTUPerNet(backbone,num_classes=configs['num_classes'],decoder_segmentation_channels=configs['decoder_segmentation_channels'],pyramid_channels=configs['pyramid_channels'],activation=None, output_size=(img_size, img_size))
    elif configs["architecture"].lower() == "deeplab":
        model = smp.DeepLabV3Plus(
            encoder_name=configs["backbone"],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet"
            if configs["pretrained"]
            else None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=configs["in_channels"],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=configs["num_classes"],  # model output channels (number of classes in your dataset)
        )
    elif configs["architecture"].lower() == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=configs["backbone"],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet"
            if configs["pretrained"]
            else None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=configs["in_channels"],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=configs["num_classes"],  # model output channels (number of classes in your dataset)
        )
    elif configs["architecture"].lower() == "upernet":
        model = upernet.UperNet(configs)
    elif configs["architecture"].lower() == "segformer":
        model = segformer.Segformer(configs)
    elif configs["architecture"].lower() == "maskedresnet_unet":
        from model_zoo.MaskedResNetUNet import MaskedResNetUNet
        model = MaskedResNetUNet(
            in_channels=configs["in_channels"],
            num_classes=configs["num_classes"],
            pretrained=configs.get("pretrained", True),
            retain_percent=configs.get("retain_percent", 100),
            mask_threshold=configs.get("mask_threshold", 0.9),
            implementation=configs.get("implementation", "topx%"),
            pad_value=configs.get("pad_value", 0.0),
        )
    else:
        print("Model not supported.")
        exit(3)
    return model


class FinetunerSegmentation(nn.Module):
    def __init__(self, encoder, configs=None, pool=False):
        super().__init__()
        self.configs = configs
        self.model = encoder
        self.pool = pool
        img_size = self.configs.get("effective_image_size", self.configs["image_size"])
        
        if not self.pool:
            if configs["deconv_decoder"]:

                self.head = nn.Sequential(
                    
                    nn.Conv2d(
                        encoder.mlp_head.in_features * (configs["in_channels"]), 512, kernel_size=3, stride=2, padding=1
                    ),
                    nn.ReLU(),
                    nn.Upsample(size=(img_size//2, img_size//2), mode="bilinear", align_corners=False),
                    nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, configs["num_classes"], kernel_size=1),
                )
            else:
                self.head = nn.Conv2d(
                    encoder.mlp_head.in_features * (configs["in_channels"]), configs["num_classes"], kernel_size=1
                )
        else:
            self.head = nn.Linear(
                encoder.mlp_head.in_features * (configs["in_channels"]),
                configs["num_classes"] * img_size * img_size,
            )
        self.model.mlp_head = nn.Identity()

    def forward(self, x):
        x, keys = x

        img_size = self.configs.get("effective_image_size", self.configs["image_size"])
        GS = img_size // self.configs["finetuning_patch_size"]
        
        x = self.model((x, keys), pool=self.pool)
        if self.pool == False:
            x = einops.rearrange(x, "b (k h w) c -> b (c k) h w", h=GS, w=GS, k=len(keys))
            
        x = self.head(x)
        
        return x


def get_encoder_head_in_features(encoder):
    """Get in_features of the encoder's classifier head (RViT/timm uses model.head, MAE-style uses mlp_head)."""
    if hasattr(encoder, "mlp_head") and encoder.mlp_head is not None:
        return encoder.mlp_head.in_features
    if hasattr(encoder, "model") and hasattr(encoder.model, "head") and encoder.model.head is not None:
        return encoder.model.head.in_features
    raise AttributeError(
        f"Encoder {type(encoder).__name__} has no classifier head (expected mlp_head or model.head)."
    )


class FinetunerClassification(nn.Module):
    def __init__(self, encoder, configs=None, pool=True):
        super().__init__()
        self.configs = configs
        self.model = encoder
        self.pool = pool
        in_features = get_encoder_head_in_features(encoder)
        self.head = nn.Linear(in_features, configs["num_classes"])
        if hasattr(encoder, "mlp_head"):
            self.model.mlp_head = nn.Identity()
        elif hasattr(encoder, "model") and hasattr(encoder.model, "head"):
            self.model.model.head = nn.Identity()

    def forward(self, x, **kwargs):
        x, keys = x
        # RViT.forward(x, keys=None, label=None) does not take pool; drop it if present
        kwargs = {k: v for k, v in kwargs.items() if k != "pool"}
        x = self.model((x, keys), **kwargs)
        x = self.head(x)
        return x

def create_model(configs):
    print("="*20)
    print("Creating model: task={}, architecture={}, backbone={}, device={}".format(
        configs.get("task"), configs.get("architecture"), configs.get("backbone"), configs.get("device")))
    # print(configs)  # full dump disabled; uncomment for debugging
    print("="*20)
    # Model resolution (pos_embed size): use enforce_resize when set, else dataset image_size
    effective_image_size = (
        configs["enforce_resize"]
        if configs.get("enforce_resize") is not None
        else configs["image_size"]
    )
    configs["effective_image_size"] = effective_image_size
    if configs.get("enforce_resize") is not None:
        print("Model image_size (pos_embed): {} (from enforce_resize)".format(effective_image_size))
    else:
        print("Model image_size (pos_embed): {} (from dataset config)".format(effective_image_size))
    if "fully_finetune" not in configs:
        fully_finetune = False
    else:
        fully_finetune = configs["fully_finetune"]

    if not fully_finetune and not configs["linear_evaluation"]:
        if configs["task"] == "classification":
            model = create_classifier(configs=configs)
        elif configs["task"] == "segmentation":
            model = create_segmentor(configs=configs)
        else:
            print("Model for task: ", configs["task"], " is not implemented yet.")
            exit(3)

    # Skip loading from resume_checkpoint_path during test phase - we load from eval_checkpoint separately
    if (configs["resume_checkpoint_path"] is not None and 
        not (configs["linear_evaluation"] or fully_finetune) and 
        configs.get("phase", "train") != "test"):
        print("Loading checkpoint from: ", configs["resume_checkpoint_path"])
        model.load_state_dict(torch.load(configs["resume_checkpoint_path"], map_location="cpu"))

        # checkpoint = torch.load(configs["resume_checkpoint_path"], map_location="cpu")
        
        # # If checkpoint is a full model object, extract state_dict
        # if not isinstance(checkpoint, dict) or "state_dict" in checkpoint:
        #     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        #         checkpoint = checkpoint["state_dict"]
        #     else:
        #         # It's a full model, try to get state_dict
        #         try:
        #             checkpoint = checkpoint.state_dict()
        #         except:
        #             checkpoint = checkpoint
        
        # # Filter out classification head if number of classes differs
        # model_state = model.state_dict()
        # filtered_checkpoint = {}
        # for key, value in checkpoint.items():
        #     # Skip classification head if sizes don't match
        #     if "head" in key.lower() or "classifier" in key.lower() or "fc" in key.lower():
        #         if key in model_state and model_state[key].shape != value.shape:
        #             print(f"  Skipping {key}: shape mismatch (checkpoint: {value.shape}, model: {model_state[key].shape})")
        #             continue
        #     filtered_checkpoint[key] = value
        
        # # Load with strict=False to allow partial loading
        # missing_keys, unexpected_keys = model.load_state_dict(filtered_checkpoint, strict=False)
        # if missing_keys:
        #     print(f"  Missing keys (will use random init): {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  Missing keys: {missing_keys}")
        # if unexpected_keys:
        #     print(f"  Unexpected keys (ignored): {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  Unexpected keys: {unexpected_keys}")
    
    if configs["linear_evaluation"] or fully_finetune:
        transfer_seg_ckpt = configs.get("segmentation_pretrained_checkpoint") or configs.get("transfer_from_segmentation_checkpoint")
        use_segmentation_pretrained = (
            configs["task"] == "classification"
            and configs.get("seg_to_class", False)
            and transfer_seg_ckpt is not None
        )

        if configs["pretrained_model_path"] is not None:
            print("Loading checkpoint from: ", configs["pretrained_model_path"])
            # Classification: create model for *target* config (e.g. MLRSNet), load encoder from
            # pretrained checkpoint (e.g. BigEarthNet) with channel/resolution adaptation, then freeze and wrap.
            # Segmentation: load full model from checkpoint (existing behavior).
            if configs["task"] == "classification":
                model = create_classifier(configs=configs)
                source_rgb_indices = configs.get("transfer_source_rgb_indices")
                load_classification_encoder_into_classifier(
                    model, configs["pretrained_model_path"], strict=False, source_rgb_indices=source_rgb_indices
                )
            elif configs["task"] == "segmentation":
                model = torch.load(configs["pretrained_model_path"], map_location="cpu")
            else:
                print("Finetuning on ", configs["task"], " is not yet supported!")
                exit(3)
        elif use_segmentation_pretrained:
            # Linear evaluation (or full finetune) on classification using encoder from segmentation (e.g. FLAIR).
            # create_classifier with seg_to_class + segmentation_pretrained_checkpoint loads the encoder.
            print("Loading encoder from segmentation checkpoint (e.g. FLAIR) for linear evaluation: ", transfer_seg_ckpt)
            model = create_classifier(configs=configs)
        else:
            print("For linear evaluation or finetuning, set pretrained_model_path or (seg_to_class + segmentation_pretrained_checkpoint). Exiting!")
            exit(2)

        if configs["pretrained_model_path"] is not None or use_segmentation_pretrained:
            if configs["linear_evaluation"]:
                print("Linear evaluation: Freezing encoder; only the classification head will be trained.")
                for param in model.parameters():
                    param.requires_grad = False
            elif fully_finetune:
                print("Fully finetuning: Enable gradients for all layers.")
                for param in model.parameters():
                    param.requires_grad = True

            if hasattr(model, "cls_token") and model.cls_token is not None:
                model.cls_token.requires_grad = True

            if configs.get("change_finetuning_resolution") is not None:
                patch_size = configs["finetuning_patch_size"]
                num_patches = (configs["change_finetuning_resolution"] // patch_size) * (
                    configs["change_finetuning_resolution"] // patch_size
                )
                assert (
                    configs["change_finetuning_resolution"] % patch_size == 0
                ), "Image dimensions must be divisible by the patch size."
                if hasattr(model, "model") and hasattr(model.model, "pos_embed"):
                    old_pe = model.model.pos_embed
                    model.model.pos_embed = nn.Parameter(
                        torch.randn(1, num_patches + 1, old_pe.shape[-1])
                    )
                    print("New positional embeddings: ", model.model.pos_embed.shape)
                elif hasattr(model, "pos_embedding"):
                    model.pos_embedding = nn.Parameter(
                        torch.randn(1, num_patches + 1, model.pos_embedding.shape[-1])
                    )
                    if hasattr(model, "num_patches"):
                        model.num_patches = num_patches
                    print("New positional embeddings: ", model.pos_embedding.shape)

            if configs["task"] == "classification":
                model = FinetunerClassification(model, configs)
            elif configs["task"] == "segmentation":
                print("=" * 20)
                print("Finetuning FoMo-Net for Segmentation")
                print("=" * 20)
                model = FinetunerSegmentation(model, configs)

            if configs.get("resume_checkpoint_path") is not None:
                if 'full_ckpt' in configs and configs['full_ckpt']:
                    print("Resuming from FoMo checkpoint: ", configs["resume_checkpoint_path"])
                    # model = torch.load(configs["resume_checkpoint_path"], map_location="cpu")
                    #model.load_state_dict(torch.load(configs["resume_checkpoint_path"], map_location="cpu"))
                    model = torch.load(configs["resume_checkpoint_path"], map_location="cpu")
                else:
                    print("Resuming from FoMo checkpoint: ", configs["resume_checkpoint_path"])
                    # model = torch.load(configs["resume_checkpoint_path"], map_location="cpu")
                    model.load_state_dict(torch.load(configs["resume_checkpoint_path"], map_location="cpu"))

    return model


def adjust_learning_rate(optimizer, epoch, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if configs['warmup_epochs']==0:
        configs['warmup_epochs']=-1
    if epoch <= configs["warmup_epochs"]:
        lr = configs["lr"] * epoch / configs["warmup_epochs"]
    else:
        lr = configs["min_lr"] + (configs["lr"] - configs["min_lr"]) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - configs["warmup_epochs"]) / (configs["epochs"] - configs["warmup_epochs"]))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_current_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
