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
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)

model_channels = {
    "vit_large_patch16_224": [1024, 768, 384, 192],
    "vit_base_patch16_224": [768, 512, 256, 128],
    "vit_small_patch16_224": [384, 256, 128, 64],
    "vit_tiny_patch16_224": [192, 128, 64, 32],
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

class RViTUPerNet(nn.Module):
    def __init__(self, encoder, num_classes=21,decoder_segmentation_channels=64,pyramid_channels=256,activation=None,output_size=(128,128)):
        super(RViTUPerNet, self).__init__()
        self.encoder = encoder  # Use RViT as the encoder
        self.plot_img = True
        self.plot_cnt = 0
        self.decoder = UPerNetDecoder(
            encoder_channels=model_channels[encoder.backbone],  # Adjust according to RViT's output feature sizes
            pyramid_channels=pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            output_size=output_size,
        )
        #Calculate dimensions depending on image size and feature sizes
        height = int(encoder.model.patch_embed.img_size[0]//encoder.model.patch_embed.patch_size[0])
        feature_size = encoder.model.patch_embed.proj.out_channels
        num_tokens = encoder.model.patch_embed.num_patches
        self.height = height
        shapes =[closest_divisible_by_2(height), closest_divisible_by_2(height//2),closest_divisible_by_2(height//4), closest_divisible_by_2(height//8) ]#height//4, height//8]
        self.shapes = shapes

        reshaped_features = []
        #Calculate reshaped features
        for i in range(4):
            resized_tokens = shapes[i]*shapes[i]
            reshaped_features.append(int((num_tokens/resized_tokens)*feature_size))            
        
        #print(self.shapes)

        #print("reshaped features: ",reshaped_features)
        self.proj1 =  nn.Conv2d(reshaped_features[0], model_channels[encoder.backbone][0], kernel_size=1, stride=1, padding=0)
        self.proj2 = nn.Conv2d(reshaped_features[1], model_channels[encoder.backbone][1], kernel_size=1, stride=1, padding=0)
        self.proj3 = nn.Conv2d(reshaped_features[2], model_channels[encoder.backbone][2], kernel_size=1, stride=1, padding=0)
        self.proj4 = nn.Conv2d(reshaped_features[3], model_channels[encoder.backbone][3], kernel_size=1, stride=1, padding=0)
        
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







    def forward(self, x, label=None):
        
        raw_input = x.clone()
        # Extract multi-scale features from the encoder
        features, mask, final_mask = self.encoder(x,label)    
         
        height = width = 0
        decoder_list = []
        for f in range(len(features)):
            height = width = self.shapes[f]
            if self.training:
                decoder_tokens = torch.zeros((features[f].shape[0],mask.shape[1],features[f].shape[2]),device=features[f].device)
                decoder_tokens = decoder_tokens.masked_scatter(mask.unsqueeze(-1),features[f])
            else:
                decoder_tokens = features[f]
            decoder_tokens = decoder_tokens.permute(0, 2, 1).contiguous().view(decoder_tokens.shape[0],-1, height, width)
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
            #Reshape mask to patch size x patch size
            mask = mask.view(mask.shape[0], self.height, self.height).float().unsqueeze(1)
            #Return the mask to the original size
            mask = F.interpolate(mask, size=(out.shape[2], out.shape[3]), mode="nearest").squeeze()
            return out, mask.bool()
        else:
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

    """
            Visualize raw input, masked image/label, and prediction.
            Args:
                raw_input: The original input tensor.
                label: The ground truth label.
                final_mask: The binary mask used in masking.
                pred: The final prediction (e.g., from segmentation head).
                patch_size: The size of each patch.
                num_patches_w: Number of patches along the image width.
            
            
            # Convert raw label to NumPy array
            raw_label = label[0].cpu().detach().numpy()
            # Use pred that was computed in your segmentation head
            pred_vis = pred[0].cpu().detach().numpy()
            #print("Prediction shape:", pred_vis.shape)
            
            # Prepare raw image for visualization
            raw_img = raw_input[0].detach().cpu().numpy()
            if raw_img.shape[0] > 3:
                raw_img = raw_img[:3, :, :]
            raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
            raw_img = raw_img.transpose(1, 2, 0)
            
            # Create masked image and masked label by mapping patches back to image space
            masked_img = np.zeros_like(raw_img)
            masked_label = np.zeros_like(raw_label)
            patch_mask = final_mask[0].cpu().numpy()  # Using the first sample's mask
            total_patches = final_mask.shape[1]
            
            for idx in range(total_patches):
                h_idx = idx // num_patches_w
                w_idx = idx % num_patches_w

                h_start = h_idx * patch_size
                h_end = (h_idx + 1) * patch_size
                w_start = w_idx * patch_size
                w_end = (w_idx + 1) * patch_size
                
                if patch_mask[idx]:
                    masked_img[h_start:h_end, w_start:w_end] = raw_img[h_start:h_end, w_start:w_end]
                    masked_label[h_start:h_end, w_start:w_end] = raw_label[h_start:h_end, w_start:w_end]
                else:
                    masked_img[h_start:h_end, w_start:w_end] = 0.0
                    masked_label[h_start:h_end, w_start:w_end] = -1.0

            # Plot side by side
            fig, axes = plt.subplots(1, 4, figsize=(16, 6))
            axes[0].imshow(raw_img)
            axes[0].set_title("Raw Input")
            axes[0].axis("off")
            
            axes[1].imshow(masked_img)
            axes[1].set_title("Masked Image")
            axes[1].axis("off")
            
            axes[2].imshow(masked_label)
            axes[2].set_title("Masked Label")
            axes[2].axis("off")
            
            axes[3].imshow(pred_vis)
            axes[3].set_title("Model Prediction")
            axes[3].axis("off")
            
            plt.tight_layout()
            plt.close()

            if wandb.run:
                wandb.log({
                    "Masked vs Raw": wandb.Image(fig),
                    "Retained Patches": patch_mask.sum() / total_patches
                })

            

            # Here we assume that 'mask' from the encoder is your final_mask.
            #visualization(raw_input, label, mask, pred, patch_size, num_patches_w) 
        """