from segmentation_models_pytorch.base import modules as md
from model_zoo.RViT import RViT
#from RViT import RViT
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import wandb


def _deterministic_adaptive_avg_pool2d(x: torch.Tensor, output_size) -> torch.Tensor:
    """
    Deterministic replacement for F.adaptive_avg_pool2d.
    Uses AvgPool2d (deterministic backward) or mean instead of AdaptiveAvgPool2d
    (non-deterministic backward per PyTorch issue #49314).
    """
    oh, ow = (output_size, output_size) if isinstance(output_size, int) else output_size
    _, _, H, W = x.shape
    if oh == 1 and ow == 1:
        return x.mean(dim=(2, 3), keepdim=True)
    if H % oh == 0 and W % ow == 0:
        stride_h, stride_w = H // oh, W // ow
        kernel_h = H - (oh - 1) * stride_h
        kernel_w = W - (ow - 1) * stride_w
        return F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))
    stride_h = max(1, (H - 1) // oh + 1) if oh > 1 else H
    stride_w = max(1, (W - 1) // ow + 1) if ow > 1 else W
    kernel_h = H - (oh - 1) * stride_h if oh > 1 else H
    kernel_w = W - (ow - 1) * stride_w if ow > 1 else W
    kernel_h, kernel_w = max(1, kernel_h), max(1, kernel_w)
    return F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))


class DeterministicAdaptiveAvgPool2d(nn.Module):
    """Deterministic replacement for nn.AdaptiveAvgPool2d (non-deterministic backward)."""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return _deterministic_adaptive_avg_pool2d(x, self.output_size)


from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)

model_channels = {
    "vit_large_patch16_224": [1024, 768, 384, 192],
    "vit_base_patch16_224": [768, 512, 256, 128],
    "vit_base_patch32_224": [768, 512, 256, 128],  # same embed_dim as vit_base_patch16_224
    "vit_base_patch8_224": [768, 512, 256, 128],  # same embed_dim as vit_base_patch16_224
    "vit_small_patch16_224": [384, 256, 128, 64],
    "vit_tiny_patch16_224": [192, 128, 64, 32],
    # DINOv2 (patch14, pretrained on LVD-142M); use with backbone e.g. vit_base_patch14_dinov2.lvd142m
    "vit_small_patch14_dinov2": [384, 256, 128, 64],
    "vit_base_patch14_dinov2": [768, 512, 256, 128],
    "vit_large_patch14_dinov2": [1024, 768, 384, 192],
    "vit_giant_patch14_dinov2": [1408, 1024, 512, 256],
    # DINOv2 with 4 register tokens (e.g. vit_base_patch14_reg4_dinov2.lvd142m)
    "vit_small_patch14_reg4_dinov2": [384, 256, 128, 64],
    "vit_base_patch14_reg4_dinov2": [768, 512, 256, 128],
    "vit_large_patch14_reg4_dinov2": [1024, 768, 384, 192],
    "vit_giant_patch14_reg4_dinov2": [1408, 1024, 512, 256],
}


class PSPModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        sizes=(1, 2, 3, 6),
        use_batchnorm=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    #DeterministicAdaptiveAvgPool2d(size),
                    md.Conv2dReLU(
                        in_channels,
                        in_channels // len(sizes),
                        kernel_size=1,
                        use_batchnorm=use_batchnorm,
                    ),
                )
                for size in sizes
            ]
        )
        self.out_conv = md.Conv2dReLU(
            in_channels=in_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            use_batchnorm=True,
        )

    def forward(self, x):
        _, _, height, width = x.shape
        out = [x] + [
            F.interpolate(
                block(x), size=(height, width), mode="bilinear", align_corners=False
            )
            for block in self.blocks
        ]
        out = self.out_conv(torch.cat(out, dim=1))
        return out


class FPNBlock(nn.Module):
    def __init__(self, skip_channels, pyramid_channels, use_bathcnorm=True):
        super().__init__()
        self.skip_conv = (
            md.Conv2dReLU(
                skip_channels,
                pyramid_channels,
                kernel_size=1,
                use_batchnorm=use_bathcnorm,
            )
            if skip_channels != 0
            else nn.Identity()
        )

    def forward(self, x, skip):
        _, channels, height, width = skip.shape
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        if channels != 0:
            skip = self.skip_conv(skip)
            x = x + skip
        return x


class UPerNetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=64,
        output_size=(128, 128),
    ):
        super().__init__()
        self.output_size = output_size
        if encoder_depth < 3:
            raise ValueError(
                "Encoder depth for UPerNet decoder cannot be less than 3, got {}.".format(
                    encoder_depth
                )
            )

        encoder_channels = encoder_channels[::-1]

        # PSP Module
        self.psp = PSPModule(
            in_channels=encoder_channels[0],
            out_channels=pyramid_channels,
            sizes=(1, 2, 3, 6),
            use_batchnorm=True,
        )

        # FPN Module
        self.fpn_stages = nn.ModuleList(
            [FPNBlock(ch, pyramid_channels) for ch in encoder_channels[1:]]
        )

        self.fpn_bottleneck = md.Conv2dReLU(
            in_channels=(len(encoder_channels) - 1) * pyramid_channels,
            out_channels=segmentation_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

    def forward(self, *features):
        output_size = self.output_size#features[0].shape[2:]
        target_size = [size // 4 for size in output_size]

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        psp_out = self.psp(features[0])

        fpn_features = [psp_out]
        for feature, stage in zip(features[1:], self.fpn_stages):
            fpn_feature = stage(fpn_features[-1], feature)
            fpn_features.append(fpn_feature)

        # Resize all FPN features to 1/4 of the original resolution.
        resized_fpn_features = []
        for feature in fpn_features:
            resized_feature = F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            resized_fpn_features.append(resized_feature)

        output = self.fpn_bottleneck(torch.cat(resized_fpn_features, dim=1))

        return output


def size_mb(tensor):
    # Calculate the number of elements in the tensor
    num_elements = tensor.numel()
    dtype = tensor.dtype    
    # Define bytes per element based on dtype (4 bytes for float32)
    bytes_per_element = 4 if dtype == torch.float32 else 8  # Add more cases if needed

    # Calculate the size in MB
    size_in_mb = (num_elements * bytes_per_element) / (1024 ** 2)

    print(f"Size of tensor in MB: {size_in_mb:.2f} MB")


def closest_divisible_by_2(num):
    # Round to the nearest even number
    return num if num % 2 == 0 else num + 1 if num > 0 else num - 1

def _get_encoder_channels(backbone_name):
    """Resolve encoder channel list by backbone name; supports timm variant suffixes (e.g. .augreg2_in21k_ft_in1k)."""
    if backbone_name in model_channels:
        return model_channels[backbone_name]
    base_name = backbone_name.split(".")[0]
    if base_name in model_channels:
        return model_channels[base_name]
    raise KeyError(f"Unknown backbone '{backbone_name}'. Add it to model_channels in RViTUPerNet.py (base: '{base_name}').")


class RViTUPerNet(nn.Module):
    def __init__(self, encoder, num_classes=21,decoder_segmentation_channels=64,pyramid_channels=256,activation=None,output_size=(128,128)):
        super(RViTUPerNet, self).__init__()
        self.encoder = encoder  # Use RViT as the encoder
        self.plot_img = True
        self.plot_cnt = 0
        channels = _get_encoder_channels(encoder.backbone)
        self.decoder = UPerNetDecoder(
            encoder_channels=channels,  # Adjust according to RViT's output feature sizes
            pyramid_channels=pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            output_size=output_size,
        )
        #Calculate dimensions depending on image size and feature sizes
        height = int(encoder.model.patch_embed.img_size[0]//encoder.model.patch_embed.patch_size[0])
        feature_size = encoder.model.patch_embed.proj.out_channels
        num_tokens = encoder.model.patch_embed.num_patches
        self.height = height
        shapes = [closest_divisible_by_2(height), closest_divisible_by_2(height//2), closest_divisible_by_2(height//4), closest_divisible_by_2(height//8)]
        # Ensure each pyramid level has at least 1x1 to avoid division by zero and invalid interpolate (e.g. enforce_resize=64)
        shapes = [max(1, s) for s in shapes]
        self.shapes = shapes

        reshaped_features = []
        #Calculate reshaped features
        for i in range(4):
            resized_tokens = shapes[i] * shapes[i]
            reshaped_features.append(int((num_tokens/resized_tokens)*feature_size))
        self.reshaped_features = reshaped_features  # for token-grid normalization (e.g. DINOv2 37x37 -> 36x36)
        
        #print(self.shapes)

        #print("reshaped features: ",reshaped_features)
        self.proj1 =  nn.Conv2d(reshaped_features[0], channels[0], kernel_size=1, stride=1, padding=0)
        self.proj2 = nn.Conv2d(reshaped_features[1], channels[1], kernel_size=1, stride=1, padding=0)
        self.proj3 = nn.Conv2d(reshaped_features[2], channels[2], kernel_size=1, stride=1, padding=0)
        self.proj4 = nn.Conv2d(reshaped_features[3], channels[3], kernel_size=1, stride=1, padding=0)
        
        self.proj = nn.Sequential(self.proj1, self.proj2, self.proj3, self.proj4)

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_segmentation_channels,
            out_channels=num_classes,
            activation=activation,
            kernel_size=1,
            upsampling=4,
        )

    def visualization_output(self, raw_input, raw_label, patch_size, num_patches_w, total_patches, pred):
    #Convert raw input to CPU numpy array
        raw_img = raw_input[0].detach().cpu().numpy()  # take first sample
        if raw_img.shape[0] > 3:
            raw_img = raw_img[:3, :, :]
        raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
        raw_img = raw_img.transpose(1, 2, 0)

        # Convert raw_label to CPU numpy array
        raw_label = raw_label[0].detach().cpu().numpy()


        for idx in range(total_patches):
            h_idx = idx // num_patches_w
            w_idx = idx % num_patches_w
            
            h_start = h_idx * patch_size
            h_end = (h_idx + 1) * patch_size
            w_start = w_idx * patch_size
            w_end = (w_idx + 1) * patch_size

        # Plot side by side
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        axes[0].imshow(raw_img)
        axes[0].set_title("Raw Input")
        axes[0].axis("off")
        
        axes[1].imshow(raw_label)
        axes[1].set_title(f"Raw Label")
        axes[1].axis("off")

        axes[2].imshow(pred)
        axes[2].set_title(f"Pred Label")
        axes[2].axis("off")

        plt.tight_layout()
        plt.close()

        if wandb.run:
            wandb.log({
                "Raw Input vs Raw Label vs Pred Label": wandb.Image(fig),
                #"Retained Patches": patch_mask.sum() / total_patches
            })

    def _build_scatter_source_for_mask(self, features_stage, mask):
        """Build scatter-friendly source for masked_scatter when encoder uses retention (topx% or max_len).
        
        masked_scatter iterates over mask True positions in row-major (batch-first) order and assigns
        sequential elements from source. With max_len, features_stage is [B, max_len, C] but valid tokens
        per batch vary (lens[b] = mask[b].sum()). Flattening features_stage naively would interleave
        padding from batch 0 into batch 1's tokens, corrupting the scatter and collapsing performance.
        
        This builds source as [sum(lens), C] with tokens in correct order: batch 0's lens[0] tokens,
        then batch 1's lens[1] tokens, etc. Works for both topx% (fixed k) and max_len (variable lens).
        """
        B, seq_len, C = features_stage.shape
        lens = mask.sum(dim=1)  # [B] number of retained tokens per batch
        total_retained = lens.sum().item()
        
        scatter_parts = []
        for b in range(B):
            scatter_parts.append(features_stage[b, :lens[b], :])  # [lens[b], C]
        return torch.cat(scatter_parts, dim=0)  # [total_retained, C]

    def _normalize_token_grid_to_expected(self, first_stage_tokens):
        """When encoder token count != height^2 (e.g. DINOv2 37x37=1369 vs expected 36x36=1296),
        interpolate to expected grid and build 4 decoder stages from the single normalized map."""
        B, num_tokens, C = first_stage_tokens.shape
        h = w = int(round(num_tokens ** 0.5))
        if h * w != num_tokens:
            raise RuntimeError("Token count {} is not a perfect square.".format(num_tokens))
        # (B, C, h, w) e.g. (B, 768, 37, 37) -> interpolate to (B, 768, 36, 36)
        x = first_stage_tokens.permute(0, 2, 1).contiguous().view(B, C, h, w)
        x = F.interpolate(x, size=(self.height, self.height), mode="bilinear", align_corners=False)
        out0 = self.proj[0](x)
        # Stage 1: 2x2 rearrange (B, C, 36, 36) -> (B, 3072, 18, 18)
        s1 = self.shapes[1]
        x_18 = x.view(B, C, 2, s1, 2, s1).permute(0, 1, 2, 4, 3, 5).contiguous().view(B, self.reshaped_features[1], s1, s1)
        out1 = self.proj[1](x_18)
        # Stage 2: interpolate, repeat channels to match reshaped_features[2]
        s2, s3 = self.shapes[2], self.shapes[3]
        x_10 = F.interpolate(x, size=(s2, s2), mode="bilinear", align_corners=False)
        n2 = self.reshaped_features[2]
        repeat2 = (n2 + C - 1) // C
        x_10 = x_10.repeat(1, repeat2, 1, 1)[:, :n2]
        out2 = self.proj[2](x_10)
        # Stage 3: interpolate to 4x4, repeat channels
        x_4 = F.interpolate(x, size=(s3, s3), mode="bilinear", align_corners=False)
        n3 = self.reshaped_features[3]
        repeat3 = (n3 + C - 1) // C
        x_4 = x_4.repeat(1, repeat3, 1, 1)[:, :n3]
        out3 = self.proj[3](x_4)
        return [out0, out1, out2, out3]




    def forward(self, x, label=None, keys=None, return_mask=False):
        
        raw_input = x.clone()
        # Extract multi-scale features from the encoder
        # Pass keys to encoder if provided (needed for random_topx% implementation)
        features, mask, final_mask = self.encoder(x, keys=keys, label=label)
         
        height = width = 0
        decoder_list = []
        
        # Check if masking was applied during inference (features have fewer tokens than mask indicates)
        is_masked_inference = (not self.training and mask is not None and 
                              features[0].shape[1] < mask.shape[1])
        
        for f in range(len(features)):
            height = width = self.shapes[f]
            
            if self.training:
                # Training mode: restore full spatial layout using mask
                # All hierarchical stages have the same token count (no downsampling in blocks)
                # So we restore all stages to the original token count using mask.shape[1]
                decoder_tokens = torch.zeros((features[f].shape[0], mask.shape[1], features[f].shape[2]), device=features[f].device, dtype=features[f].dtype)
                scatter_source = self._build_scatter_source_for_mask(features[f], mask)
                decoder_tokens = decoder_tokens.masked_scatter(mask.unsqueeze(-1), scatter_source)
            elif is_masked_inference:
                # Inference with masking: restore full spatial layout same as training
                decoder_tokens = torch.zeros((features[f].shape[0], mask.shape[1], features[f].shape[2]), device=features[f].device, dtype=features[f].dtype)
                scatter_source = self._build_scatter_source_for_mask(features[f], mask)
                decoder_tokens = decoder_tokens.masked_scatter(mask.unsqueeze(-1), scatter_source)
            else:
                # Standard inference without masking: use features directly
                decoder_tokens = features[f]
            
            # Reshape to spatial layout: [B, tokens, features] -> [B, features, H, W]
            num_tokens = decoder_tokens.shape[1]
            expected_tokens = self.height * self.height
            if num_tokens != expected_tokens:
                # Normalize token grid (e.g. DINOv2 patch14 @ 512 gives 37x37=1369; decoder expects 36x36=1296)
                decoder_list = self._normalize_token_grid_to_expected(decoder_tokens)
                break
            height = width = self.shapes[f]
            decoder_tokens = decoder_tokens.permute(0, 2, 1).contiguous().view(decoder_tokens.shape[0], -1, height, width)
            decoder_tokens = self.proj[f](decoder_tokens)
            decoder_list.append(decoder_tokens)
        
        # Pass the features to the UPerNet decoder
        out = self.decoder(decoder_list[0], decoder_list[1], decoder_list[2], decoder_list[3])
        out = self.segmentation_head(out)

        # Call visualization using the right values.
        if self.plot_img and not self.training and self.plot_cnt < 10:
            pred = torch.argmax(out.detach().cpu(), axis=1)
            self.plot_cnt += 1
            # Assume patch_size is stored in the patch embedding
            #patch_size = self.model.patch_embed.patch_size[0]
            patch_size = self.encoder.model.patch_embed.patch_size[0]
            # Calculate the number of patches along width. If your input isn't padded,
            # you might need to compute it as follows:
            num_patches_w = (raw_input.shape[3] + (patch_size - raw_input.shape[3] % patch_size) % patch_size) // patch_size

            #self.visualization_output(raw_input, label, patch_size, num_patches_w, final_mask.shape[1], pred[0].detach().cpu().numpy())
            
        if self.training:
            # #Reshape mask to patch size x patch size
            # mask = mask.view(mask.shape[0], self.height, self.height).float().unsqueeze(1)
            # #Return the mask to the original size
            # mask = F.interpolate(mask, size=(out.shape[2], out.shape[3]), mode="nearest").squeeze()
            B = mask.shape[0]
            num_tokens = mask.shape[1]
            h = w = int(round(num_tokens ** 0.5))
            mask = mask.view(B, h, w).float().unsqueeze(1)
            if h != self.height:
                mask = F.interpolate(mask, size=(self.height, self.height), mode="nearest")
            mask = mask.squeeze(1)
            # Return the mask to the original output size
            mask = F.interpolate(mask.unsqueeze(1), size=(out.shape[2], out.shape[3]), mode="nearest").squeeze(1)
            return out, mask.bool()
        else:
            if return_mask and mask is not None:
                B, num_tokens = mask.shape[0], mask.shape[1]
                h = w = int(round(num_tokens ** 0.5))
                mask_img = mask.view(B, h, w).float().unsqueeze(1)
                if h != self.height:
                    mask_img = F.interpolate(mask_img, size=(self.height, self.height), mode="nearest")
                mask_img = mask_img.squeeze(1)
                mask_img = F.interpolate(mask_img.unsqueeze(1), size=(out.shape[2], out.shape[3]), mode="nearest").squeeze(1)
                return out, mask_img.bool()
            return out
    

if __name__ == '__main__':
    img_size=512
    model = RViT(num_classes=14, backbone='vit_base_patch16_224', image_size=img_size, num_channels=14, mask_threshold=0.5, mask_in_token_space=False, implementation="topx%", pad_value = 0.0, retain_percent=60, attention_mask=False,hierarchical_features=True)
    up = RViTUPerNet(model,num_classes=14,decoder_segmentation_channels=64,pyramid_channels=256,activation=None, output_size=(img_size,img_size))
    image = torch.randn(12, 14, img_size, img_size)
    label = torch.randint(0,19,(12,img_size, img_size))
    #Get predictions
    predictions = up(image, label)
    print(predictions.shape)
    #Get classes via argmax
    
    predictions = torch.argmax(predictions, dim=1)
    print(predictions)
