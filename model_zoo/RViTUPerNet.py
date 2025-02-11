from segmentation_models_pytorch.base import modules as md
from model_zoo.RViT import RViT
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
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
        self.decoder = UPerNetDecoder(
            encoder_channels=model_channels[encoder.backbone],  # Adjust according to RViT's output feature sizes
            pyramid_channels=pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            output_size=output_size,
        )
        #Calculate dimensions depending on image size and feature sizes
        height = int(encoder.model.patch_embed.img_size[0]//encoder.model.patch_embed.patch_size[0])
        feature_size = encoder.model.patch_embed.proj.out_channels
        print(height)
        shapes =[closest_divisible_by_2(height), closest_divisible_by_2(height//2),closest_divisible_by_2(height//4), closest_divisible_by_2(height//8) ]#height//4, height//8]
        self.shapes = shapes
        print(shapes)

        num_tokens = encoder.model.patch_embed.num_patches
        reshaped_features = []
        #Calculate reshaped features
        for i in range(4):
            resized_tokens = shapes[i]*shapes[i]
            reshaped_features.append(int((num_tokens/resized_tokens)*feature_size))            
        
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

    def forward(self, x):
        # Extract multi-scale features from the encoder
        print(x.shape)
        features = self.encoder(x)
        height = width = 0
        for f in range(len(features)):
            height = width = self.shapes[f]
            print(features[f].shape)

            features[f] = features[f].permute(0, 2, 1).contiguous().view(features[f].shape[0],-1, height, width)
            features[f] = self.proj[f](features[f])
        # Pass the features to the UPerNet decoder
        out = self.decoder(features[0], features[1], features[2], features[3])
        out = self.segmentation_head(out)
        return out



if __name__ == '__main__':
    img_size=512
    model = RViT(num_classes=14, backbone='vit_base_patch16_224', image_size=img_size, num_channels=14, mask_threshold=0.9, mask_in_token_space=False, implementation="max_len", pad_value = 0.0, retain_percent=60, attention_mask=True,hierarchical_features=True)
    up = RViTUPerNet(model,num_classes=14,decoder_segmentation_channels=64,pyramid_channels=256,activation=None, output_size=(img_size,img_size))
    image = torch.randn(2, 14, img_size, img_size)
    #Get predictions
    predictions = up(image)
    print(predictions.shape)
    #Get classes via argmax
    predictions = torch.argmax(predictions, dim=1)
    print(predictions)