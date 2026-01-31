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
                           For BigEarthNet, use [4, 3, 2] (B04=Red, B03=Green, B02=Blue per modality index).
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
    """
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif not isinstance(ckpt, dict):
        ckpt = getattr(ckpt, "state_dict", lambda: ckpt)()
    model_sd = classifier_model.state_dict()
    prefix = "encoder."
    filtered = {}
    patch_embed_key = None
    patch_embed_val = None
    for k, v in ckpt.items():
        if not k.startswith(prefix):
            continue
        new_k = k[len(prefix) :]
        if new_k not in model_sd:
            continue
        target = model_sd[new_k]
        if v.shape == target.shape:
            filtered[new_k] = v
        elif new_k == "model.patch_embed.proj.weight" and v.dim() == 4:
            # Segmentation: 3 ch (e.g. FLAIR), classifier: more ch (e.g. BigEarthNet 14)
            if v.shape[0] == target.shape[0] and v.shape[2:] == target.shape[2:] and v.shape[1] == 3 and target.shape[1] >= 3:
                patch_embed_key = new_k
                patch_embed_val = v
            # else skip
        # else skip (e.g. different num_classes head)
    if patch_embed_key and patch_embed_val is not None:
        target = model_sd[patch_embed_key].clone()
        target[:, :3, :, :] = patch_embed_val
        filtered[patch_embed_key] = target
        print(f"  Transferred patch_embed.proj: FLAIR RGB (3 ch) -> first 3 channels of classifier ({target.shape[1]} ch)")
    missing, unexpected = classifier_model.load_state_dict(filtered, strict=False)
    n_loaded = len(filtered)
    print(f"  Loaded {n_loaded} keys from segmentation checkpoint (encoder -> classifier backbone).")
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
    if configs['backbone'].lower() == 'act_base':
        import model_zoo.ACTViT as ACTViT
        model = ACTViT.AgglomerativeClusteringVisionTransformer(
                patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, img_size=configs['image_size'],in_chans=configs['in_channels'],num_classes=configs['num_classes'],
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), configs=configs)
    elif "rvit" in configs and configs["rvit"]:
        # Segmentation-to-classification transfer only when explicitly enabled in config (seg_to_class + checkpoint path)
        transfer_ckpt = configs.get("segmentation_pretrained_checkpoint") or configs.get("transfer_from_segmentation_checkpoint")
        if configs.get("seg_to_class", False) and transfer_ckpt:
            # Transfer from segmentation (e.g. FLAIR): create model without ImageNet, then load encoder from checkpoint
            model = RViT.RViT(
                num_classes=configs['num_classes'],
                backbone=configs['backbone'],
                image_size=configs["image_size"],
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
                image_size=configs["image_size"],
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
                    # BigEarthNet modality index: B04=4 (Red), B03=3 (Green), B02=2 (Blue)
                    if configs.get('rgb_channels_reordered', False):
                        rgb_indices = [0, 1, 2]
                    else:
                        rgb_indices = [4, 3, 2]
                elif configs.get('dataset') == 'flair' or configs.get('dataset') == 'flair2':
                    rgb_indices = [0, 1, 2]
                model = initialize_rgb_pretrained_weights(
                    model, configs['backbone'], configs["image_size"],
                    configs['num_classes'], configs["in_channels"], rgb_channel_indices=rgb_indices
                )
    else:
        # Create model with timm (it will handle pretrained conversion automatically)
        model = timm.create_model(
            configs["backbone"].lower(),
            pretrained=configs["pretrained"],
            in_chans=configs["in_channels"],
            num_classes=configs["num_classes"],
            img_size=configs["image_size"], dynamic_img_pad=True, dynamic_img_size=True,
            #patch_size=configs["patch_size"]
        )
        # Override timm's weight conversion if pretrained and multi-channel
        if configs['pretrained'] and configs["in_channels"] > 3:
            # Determine RGB channel indices based on dataset
            rgb_indices = None
            if configs.get('dataset') == 'bigearthnet':
                # BigEarthNet modality index: RGB at indices [4, 3, 2] (B04=Red, B03=Green, B02=Blue)
                if configs.get('rgb_channels_reordered', False):
                    rgb_indices = [0, 1, 2]  # Reordered: RGB first
                else:
                    rgb_indices = [4, 3, 2]  # Modality index order: B04 (Red), B03 (Green), B02 (Blue)
            elif configs.get('dataset') == 'flair' or configs.get('dataset') == 'flair2':
                # FLAIR modality index: aerial-r=0 (Red), aerial-g=1 (Green), aerial-b=2 (Blue)
                rgb_indices = [0, 1, 2]  # RGB at first 3 channels
            
            model = initialize_rgb_pretrained_weights(
                model, configs['backbone'], configs["image_size"], 
                configs['num_classes'], configs["in_channels"], rgb_channel_indices=rgb_indices
            )
        return model
    return model


def create_segmentor(configs):
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
                    rgb_indices = [4, 3, 2]  # Modality index order: B04 (Red), B03 (Green), B02 (Blue)
            elif configs.get('dataset') == 'flair' or configs.get('dataset') == 'flair2':
                # FLAIR modality index: aerial-r=0 (Red), aerial-g=1 (Green), aerial-b=2 (Blue)
                rgb_indices = [0, 1, 2]  # RGB at first 3 channels
            
            # Apply RGB initialization to the encoder (smp uses timm encoders)
            model.encoder = initialize_rgb_pretrained_weights(
                model.encoder, configs['backbone'], configs["image_size"], 
                configs['num_classes'], configs["in_channels"], rgb_channel_indices=rgb_indices
            )
    elif configs["architecture"].lower() == "rvitupernet":
        # Create backbone with pretrained (timm will handle conversion)
        backbone = RViT.RViT(
            num_classes=configs['num_classes'],
            backbone=configs['backbone'],
            image_size=configs["image_size"],
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
                # BigEarthNet modality index: RGB at indices [4, 3, 2] (B04=Red, B03=Green, B02=Blue)
                if configs.get('rgb_channels_reordered', False):
                    rgb_indices = [0, 1, 2]  # Reordered: RGB first
                else:
                    rgb_indices = [4, 3, 2]  # Modality index order: B04 (Red), B03 (Green), B02 (Blue)
            elif configs.get('dataset') == 'flair' or configs.get('dataset') == 'flair2':
                # FLAIR modality index: aerial-r=0 (Red), aerial-g=1 (Green), aerial-b=2 (Blue)
                rgb_indices = [0, 1, 2]  # RGB at first 3 channels
            
            backbone = initialize_rgb_pretrained_weights(
                backbone, configs['backbone'], configs["image_size"], 
                configs['num_classes'], configs["in_channels"], rgb_channel_indices=rgb_indices
            )
        model = RViTUPerNet.RViTUPerNet(backbone,num_classes=configs['num_classes'],decoder_segmentation_channels=configs['decoder_segmentation_channels'],pyramid_channels=configs['pyramid_channels'],activation=None, output_size=(configs['image_size'],configs['image_size']))
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
        img_size = self.configs["image_size"]
        
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
                configs["num_classes"] * configs["image_size"] * configs["image_size"],
            )
        self.model.mlp_head = nn.Identity()

    def forward(self, x):
        x, keys = x

        img_size = self.configs["image_size"]
        GS = img_size // self.configs["finetuning_patch_size"]
        
        x = self.model((x, keys), pool=self.pool)
        if self.pool == False:
            x = einops.rearrange(x, "b (k h w) c -> b (c k) h w", h=GS, w=GS, k=len(keys))
            
        x = self.head(x)
        
        return x


class FinetunerClassification(nn.Module):
    def __init__(self, encoder, configs=None, pool=True):
        super().__init__()
        self.configs = configs
        self.model = encoder
        self.pool = pool
        self.head = nn.Linear(encoder.mlp_head.in_features, configs["num_classes"])
        self.model.mlp_head = nn.Identity()

    def forward(self, x):
        x, keys = x
        x = self.model((x, keys), pool=self.pool)
        x = self.head(x)
        return x

def create_model(configs):
    print("="*20)
    print("Creating model: task={}, architecture={}, backbone={}, device={}".format(
        configs.get("task"), configs.get("architecture"), configs.get("backbone"), configs.get("device")))
    # print(configs)  # full dump disabled; uncomment for debugging
    print("="*20)
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
        if configs["pretrained_model_path"] is not None:
            print("Loading checkpoint from: ", configs["pretrained_model_path"])
            model = torch.load(configs["pretrained_model_path"], map_location="cpu")

            if configs["linear_evaluation"]:
                print("Linear evaluation: Freezing all layers except the classification head.")
                for param in model.parameters():
                    param.requires_grad = False
            elif fully_finetune:
                print("Fully finetuning: Enable gradients for all layers.")
                for param in model.parameters():
                    param.requires_grad = True

            model.cls_token.requires_grad = True

            if configs["change_finetuning_resolution"] is not None:
                patch_size = configs["finetuning_patch_size"]
                num_patches = (configs["change_finetuning_resolution"] // patch_size) * (
                    configs["change_finetuning_resolution"] // patch_size
                )
                model.num_patches = num_patches
                assert (
                    configs["change_finetuning_resolution"] % patch_size == 0
                ), "Image dimensions must be divisible by the patch size."
                model.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, model.pos_embedding.shape[-1]))
                print("New positional embeddings: ")
                print(model.pos_embedding.shape)

            # Assume a transformer like model from MAE for now
            if configs["task"] == "classification":
                model = FinetunerClassification(model, configs)
            elif configs["task"] == "segmentation":
                print("=" * 20)
                print("Finetuning FoMo-Net for Segmentation")
                print("=" * 20)
                model = FinetunerSegmentation(model, configs)
            else:
                print("Finetuning on ", configs["task"], " is not yet supported!")

            if configs["resume_checkpoint_path"] is not None:
                if 'full_ckpt' in configs and configs['full_ckpt']:
                    print("Resuming from FoMo checkpoint: ", configs["resume_checkpoint_path"])
                    # model = torch.load(configs["resume_checkpoint_path"], map_location="cpu")
                    #model.load_state_dict(torch.load(configs["resume_checkpoint_path"], map_location="cpu"))
                    model = torch.load(configs["resume_checkpoint_path"], map_location="cpu")
                else:
                    print("Resuming from FoMo checkpoint: ", configs["resume_checkpoint_path"])
                    # model = torch.load(configs["resume_checkpoint_path"], map_location="cpu")
                    model.load_state_dict(torch.load(configs["resume_checkpoint_path"], map_location="cpu"))

        else:
            print("Pretrained model path is None! Exiting!")
            exit(2)

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
