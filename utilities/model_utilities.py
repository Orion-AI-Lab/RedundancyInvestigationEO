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
        model = RViT.RViT( num_classes=configs['num_classes'], 
            backbone=configs['backbone'], 
            image_size=configs["image_size"], 
            num_channels=configs["in_channels"], 
            pretrained=configs['pretrained'], 
            mask_threshold=configs["mask_threshold"],
            mask_in_token_space=configs['mask_in_token_space'],  
            implementation=configs['implementation'], 
            pad_value = configs['pad_value'], 
            retain_percent=configs['retain_percent'], 
            attention_mask=configs['attention_mask'],
            #patch_size=configs['patch_size'])
            #rvit=configs['rvit'])
        )
    else:
        model = timm.create_model(
            configs["backbone"].lower(),
            pretrained=configs["pretrained"],
            in_chans=configs["in_channels"],
            num_classes=configs["num_classes"],
            img_size=configs["image_size"], dynamic_img_pad=True, dynamic_img_size=True,
            #patch_size=configs["patch_size"]
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
    elif configs["architecture"].lower() == "rvitupernet":
        backbone = RViT.RViT( num_classes=configs['num_classes'], backbone=configs['backbone'], image_size=configs["image_size"], num_channels=configs["in_channels"], pretrained=configs['pretrained'], mask_threshold=configs["mask_threshold"],mask_in_token_space=configs['mask_in_token_space'],  implementation=configs['implementation'], pad_value = configs['pad_value'], retain_percent=configs['retain_percent'], attention_mask=configs['attention_mask'],hierarchical_features=True)
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
    print("Creating model with configs: ")
    print(configs)
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

    if configs["resume_checkpoint_path"] is not None and not (configs["linear_evaluation"] or fully_finetune):
        print("Loading checkpoint from: ", configs["resume_checkpoint_path"])
        model.load_state_dict(torch.load(configs["resume_checkpoint_path"], map_location="cpu"))

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
