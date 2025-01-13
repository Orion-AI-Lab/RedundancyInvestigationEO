"""
Code is mostly taken from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import einops

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels={},
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.channel_per_dataset = channels
        self.to_patch_embedding = {}

        for key in self.channel_per_dataset:
            patch_dim = self.channel_per_dataset[key] * patch_height * patch_width

            self.to_patch_embedding[key] = nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, batch):
        img, key = batch
        x = self.to_patch_embedding[key](img)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x += self.pos_embedding[:, 1 : (n + 1)]

        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class MultiModalMAE(nn.Module):
    def __init__(self, *, encoder, decoder_dim, masking_ratio=0.75, decoder_depth=1, decoder_heads=8, decoder_dim_head=64):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch = {}
        self.patch_to_emb = {}
        pixel_values_per_patch = {}
        for key in encoder.to_patch_embedding:
            self.to_patch[key] = encoder.to_patch_embedding[key][0]
            self.patch_to_emb[key] = nn.Sequential(*encoder.to_patch_embedding[key][1:])

            pixel_values_per_patch[key] = encoder.to_patch_embedding[key][2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head, mlp_dim=decoder_dim * 4
        )
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)

        self.to_pixels = {}
        for key in pixel_values_per_patch:
            self.to_pixels[key] = nn.Linear(decoder_dim, pixel_values_per_patch[key])

    def forward(self, batch):
        img, key = batch
        device = img.device

        # get patches
        self.to_patch[key].to(device)
        self.patch_to_emb[key].to(device)
        self.to_pixels[key].to(device)

        patches = self.to_patch[key](img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb[key](patches)

        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1 : (num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype)

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels[key](mask_tokens)

        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        reconstruction_data = {
            "masked_indices": masked_indices,
            "patches": patches,
            "model_input": img,
            "predicted": pred_pixel_values,
        }

        return recon_loss, decoded_tokens, reconstruction_data

    def reconstruct_image(
        self, patches, model_input, mean, std, masked_indices=None, pred_pixel_values=None, patch_size=8
    ):
        """
        Reconstruction code mostly taken from https://github.com/lucidrains/vit-pytorch/issues/169.

        Reconstructs the image given patches. Can also reconstruct the masked image as well as the predicted image.
        To reconstruct the raw image from the patches, set masked_indices=None and pred_pixel_values=None. To reconstruct
        the masked image, set masked_indices= the masked_indices tensor created in the `forward` call. To reconstruct the
        predicted image, set masked_indices and pred_pixel_values = to their respective tensors created in the `forward` call.

        ARGS:
            patches (torch.Tensor): The raw patches (pre-patch embedding) generated for the given model input. Shape is
                (batch_size x num_patches x patch_size^2 * channels)
            model_input (torch.Tensor): The input images to the given model (batch_size x channels x height x width)
            mean (list[float]): An array representing the per-channel mean of the dataset used to
                denormalize the input and predicted pixels. (1 x channels)
            std (list[float]): An array representing the per-channel std of the dataset used to
                denormalize the input and predicted pixels. (1 x channels)
            masked_indices (torch.Tensor): The patch indices that are masked (batch_size x masking_ratio * num_patches)
            pred_pixel_values (torch.Tensor): The predicted pixel values for the patches that are masked (batch_size x masking_ratio * num_patches x patch_size^2 * channels)

        RETURN:
            reconstructed_image (torch.Tensor): Tensor containing the reconstructed image (batch_size x channels x height x width)
        """

        patches = patches.cpu()

        masked_indices_in = masked_indices is not None
        predicted_pixels_in = pred_pixel_values is not None

        if masked_indices_in:
            masked_indices = masked_indices.cpu()

        if predicted_pixels_in:
            pred_pixel_values = pred_pixel_values.cpu()

        patch_width = patch_height = patch_size
        reconstructed_image = patches.clone()

        if masked_indices_in or predicted_pixels_in:
            for i in range(reconstructed_image.shape[0]):
                if masked_indices_in and predicted_pixels_in:
                    reconstructed_image[i, masked_indices[i].cpu()] = pred_pixel_values[i, :].cpu().float()
                elif masked_indices_in:
                    reconstructed_image[i, masked_indices[i].cpu()] = 0

        invert_patch = Rearrange(
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            w=int(model_input.shape[3] / patch_width),
            h=int(model_input.shape[2] / patch_height),
            c=model_input.shape[1],
            p1=patch_height,
            p2=patch_width,
        )

        reconstructed_image = invert_patch(reconstructed_image)

        reconstructed_image = reconstructed_image.numpy().transpose(0, 2, 3, 1)
        reconstructed_image *= np.array(std)
        reconstructed_image += np.array(mean)

        return reconstructed_image.transpose(0, 3, 2, 1)


class MultiSpectralViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        configs=None
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.configs = configs
        if self.configs["single_embedding_layer"]:
            self.to_patch_embedding = nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )
        else:
            self.to_patch_embedding = {}
            # TODO: Add support for different embedders
            for key in range(len(configs["modality_channels"])):
                self.to_patch_embedding[key] = nn.Sequential(
                    Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
                    nn.LayerNorm(patch_dim),
                    nn.Linear(patch_dim, dim),
                    nn.LayerNorm(dim),
                )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        num_spectral = len(configs["modality_channels"])
        # Extra channel for cls token
        self.spectral_embedding = nn.Parameter(torch.randn(1, num_spectral + 1, dim))
        self.num_spectral_patches = num_spectral
        self.total_tokens = self.num_patches * self.num_spectral_patches + 1
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, data, pool=True):
        img, keys = data
        b = img.shape[0]
        device = img.device
        img = einops.rearrange(img, "b c h w -> c b h w")
        # get patches
        tokens = None

        for idx, channel in enumerate(img):
            channel = channel.unsqueeze(1)
            if self.configs["single_embedding_layer"]:
                channel_token = self.to_patch_embedding(channel)
            else:
                self.to_patch_embedding[keys[idx]].to(device)
                channel_token = self.to_patch_embedding[keys[idx]](channel)

            channel_token += self.spectral_embedding[:, keys[idx]]
            _, n, _ = channel_token.shape
            channel_token += self.pos_embedding[:, 1 : (n + 1)]
            if tokens is None:
                tokens = channel_token
            else:
                tokens = torch.cat((tokens, channel_token), dim=1)

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b).clone()

        # cls_tokens += self.pos_embedding[:,0:1]
        # cls_tokens += self.spectral_embedding[:,-1]

        # x = torch.cat((cls_tokens, tokens), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x = tokens
        x = self.dropout(x)

        x = self.transformer(x)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        if pool:
            x = x.mean(dim=1)
            x = self.to_latent(x)
            x = self.mlp_head(x)
            return x
        else:
            # if self.pool == 'mean' else x[:, 0]

            return x
        # x = self.to_latent(x)
        # return self.mlp_head(x)


class MultiSpectralMAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio=0.75,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
        configs=None
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.num_patches = num_patches
        if configs["single_embedding_layer"]:
            self.to_patch = encoder.to_patch_embedding[0]
            self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])
            pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]
        else:
            self.to_patch = encoder.to_patch_embedding[0][0]
            self.patch_to_emb = {}
            for key in range(len(configs["modality_channels"])):
                self.patch_to_emb[key] = nn.Sequential(*encoder.to_patch_embedding[key][1:])
            pixel_values_per_patch = encoder.to_patch_embedding[0][2].weight.shape[-1]
        self.configs = configs
        self.pixel_values_per_patch = pixel_values_per_patch

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head, mlp_dim=decoder_dim * 4
        )
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        num_spectral = len(configs["modality_channels"])
        # Extra channel for cls token
        self.spectral_embedding = nn.Embedding(num_spectral, decoder_dim)
        if self.configs["single_embedding_layer"]:
            self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        else:
            self.to_pixels = {}
            for key in range(len(configs["modality_channels"])):
                self.to_pixels[key] = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, data):
        img, keys = data

        device = img.device
        batch, c, h, w = img.shape
        img = einops.rearrange(img, "b c h w -> c b h w")

        # get patches
        num_patches = 0
        tokens = None
        all_patches = None
        token_channel_dict = {}
        token_pos_dict = {}
        for idx, channel in enumerate(img):
            channel = channel.unsqueeze(1)
            if self.configs["single_embedding_layer"]:
                patches = self.to_patch(channel)
                # patch to encoder tokens and add positions

                channel_token = self.patch_to_emb(patches)
            else:
                patches = self.to_patch(channel)
                # patch to encoder tokens and add positions
                self.patch_to_emb[keys[idx]].to(device)
                channel_token = self.patch_to_emb[keys[idx]](patches)
            b, n_patches, *_ = patches.shape
            num_patches += n_patches

            # Add spectral position embedding

            channel_token += self.encoder.spectral_embedding[:, keys[idx]]
            token_channel_dict[idx] = keys[idx]
            # Add pos embedding

            if self.encoder.pool == "cls":
                channel_token += self.encoder.pos_embedding[:, 1 : (n_patches + 1)]
            elif self.encoder.pool == "mean":
                channel_token += self.encoder.pos_embedding.to(device, dtype=tokens.dtype)

            if tokens is None:
                tokens = channel_token
                all_patches = patches
            else:
                tokens = torch.cat((tokens, channel_token), dim=1)
                all_patches = torch.cat((all_patches, patches), dim=1)

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = all_patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        pos_masked_indices = masked_indices % (self.num_patches - 1)
        pos_unmasked_indices = unmasked_indices % (self.num_patches - 1)
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(pos_unmasked_indices)

        spectral_masked_indices = masked_indices // (self.num_patches - 1)
        spectral_unmasked_indices = unmasked_indices // (self.num_patches - 1)

        # Adapt the spectral indices to spectral band ids
        spectral_masked_ids = torch.zeros_like(spectral_masked_indices)
        for i in range(spectral_masked_indices.shape[0]):
            for j in range(spectral_masked_indices.shape[1]):
                spectral_masked_ids[i, j] = token_channel_dict[spectral_masked_indices[i, j].item()]

        spectral_unmasked_ids = torch.zeros_like(spectral_unmasked_indices)
        for i in range(spectral_unmasked_indices.shape[0]):
            for j in range(spectral_unmasked_indices.shape[1]):
                spectral_unmasked_ids[i, j] = token_channel_dict[spectral_unmasked_indices[i, j].item()]

        unmasked_decoder_tokens = unmasked_decoder_tokens + self.spectral_embedding(spectral_unmasked_ids)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(pos_masked_indices)
        mask_tokens = mask_tokens + self.spectral_embedding(spectral_masked_ids)
        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        if self.configs["single_embedding_layer"]:
            pred_pixel_values = self.to_pixels(mask_tokens)
        else:
            # Project to_pixel for each modality of the decoded tokens. The modality is determined by the spectral_masked_indices
            # Get unique spectral masked ids
            unique_spectral_masked_ids = torch.unique(spectral_masked_ids)
            tokens_per_id = {}
            projected = torch.zeros(
                size=(mask_tokens.shape[0], mask_tokens.shape[1], self.pixel_values_per_patch), device=device
            )
            for i in range(len(unique_spectral_masked_ids)):
                self.to_pixels[unique_spectral_masked_ids[i].item()].to(device)

                tokens_per_id[unique_spectral_masked_ids[i].item()] = mask_tokens[
                    spectral_masked_ids == unique_spectral_masked_ids[i]
                ]
                tmp_tokens = tokens_per_id[unique_spectral_masked_ids[i].item()].to(projected.dtype)
                tmp_pixel_space = self.to_pixels[unique_spectral_masked_ids[i].item()](tmp_tokens)
                projected = projected.to(tmp_pixel_space.dtype)
                projected[spectral_masked_ids == unique_spectral_masked_ids[i]] = tmp_pixel_space

            pred_pixel_values = projected

            """
            Non-Vectorized decoder
            pred_pixel_values = torch.zeros((mask_tokens.shape[0], mask_tokens.shape[1], self.pixel_values_per_patch), device=device)
            #Vectorize this
            for b in range(mask_tokens.shape[0]):
                # For each available spectral band in masked tokens feed them to the respective to_pixels layer
                for i in range(mask_tokens.shape[1]):
                    pred_pixel_values[b, i] = self.to_pixels[spectral_masked_ids[b, i]](mask_tokens[b, i])
            """
        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        reconstruction_data = {
            "masked_indices": masked_indices,
            "patches": all_patches,
            "model_input": img,
            "predicted": pred_pixel_values,
        }

        return recon_loss, decoded_tokens, reconstruction_data



class Segmentor(nn.Module):
    def __init__(self, in_features, img_size=None, in_channels=2,num_classes=2,patch_size=16):
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.in_features = in_features
        

        self.head = nn.Sequential(
            
            nn.Conv2d(
                in_features * (in_channels), 64, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Upsample(size=(img_size//2, img_size//2), mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, num_classes, kernel_size=1),
        )
        

    def forward(self, x):

        img_size = self.img_size
        GS = img_size // self.patch_size
        
        x = einops.rearrange(x, "b (k h w) c -> b (c k) h w", h=GS, w=GS, k=self.in_channels)
            
        x = self.head(x)
        
        return x
    
class SuperFomo(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio=0.75,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
        configs=None
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        self.reconstruction_loss = configs["reconstruction_loss"]
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.num_patches = num_patches
        if configs["single_embedding_layer"]:
            self.to_patch = encoder.to_patch_embedding[0]
            self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])
            pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]
        else:
            self.to_patch = encoder.to_patch_embedding[0][0]
            self.patch_to_emb = {}
            for key in range(len(configs["modality_channels"])):
                self.patch_to_emb[key] = nn.Sequential(*encoder.to_patch_embedding[key][1:])
            pixel_values_per_patch = encoder.to_patch_embedding[0][2].weight.shape[-1]
        self.configs = configs
        self.pixel_values_per_patch = pixel_values_per_patch

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head, mlp_dim=decoder_dim * 4
        )
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        num_spectral = len(configs["modality_channels"])
        # Extra channel for cls token
        self.spectral_embedding = nn.Embedding(num_spectral, decoder_dim)
        if self.configs["single_embedding_layer"]:
            self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        else:
            self.to_pixels = {}
            for key in range(len(configs["modality_channels"])):
                self.to_pixels[key] = nn.Linear(decoder_dim, pixel_values_per_patch)

        self.classification_fc = nn.ModuleDict()
        for dataset in configs["cls_train_datasets"]:
            self.classification_fc[dataset] = nn.Linear(encoder_dim, configs["total_num_classes"][dataset])
        
        self.segmentation_fc = nn.ModuleDict()

        for dataset in configs["segm_train_datasets"]:
            if configs['universal_img_size'] is None:
                img_size=configs["img_size"][dataset]
            else:
                img_size=configs["universal_img_size"]
            self.segmentation_fc = Segmentor(encoder.mlp_head.in_features, img_size=img_size, in_channels=min(configs['max_number_of_bands'],len(configs["dataset_modality_index"][dataset])), num_classes=configs["total_num_classes"][dataset], patch_size=configs["patch_size"])
        

    def forward(self, data):
        img, keys, task = data

        device = img.device
        batch, c, h, w = img.shape
        img = einops.rearrange(img, "b c h w -> c b h w")

        # get patches
        num_patches = 0
        tokens = None
        all_patches = None
        token_channel_dict = {}

        for idx, channel in enumerate(img):
            channel = channel.unsqueeze(1)
            if self.configs["single_embedding_layer"]:
                patches = self.to_patch(channel)
                # patch to encoder tokens and add positions

                channel_token = self.patch_to_emb(patches)
            else:
                patches = self.to_patch(channel)
                # patch to encoder tokens and add positions
                self.patch_to_emb[keys[idx]].to(device)
                channel_token = self.patch_to_emb[keys[idx]](patches)
            b, n_patches, *_ = patches.shape
            num_patches += n_patches

            # Add spectral position embedding

            channel_token += self.encoder.spectral_embedding[:, keys[idx]]
            token_channel_dict[idx] = keys[idx]
            # Add pos embedding

            if self.encoder.pool == "cls":
                channel_token += self.encoder.pos_embedding[:, 1 : (n_patches + 1)]
            elif self.encoder.pool == "mean":
                channel_token += self.encoder.pos_embedding.to(device, dtype=tokens.dtype)

            if tokens is None:
                tokens = channel_token
                all_patches = patches
            else:
                tokens = torch.cat((tokens, channel_token), dim=1)
                all_patches = torch.cat((all_patches, patches), dim=1)

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = all_patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)
        
        if task == "classification":
            out = self.classification_fc[task](encoded_tokens)
        elif task == "segmentation":
            out = self.segmentation_fc[task](encoded_tokens, keys)
        else:
            raise ValueError("Task not supported")


        if self.reconstruction_loss:
            # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

            decoder_tokens = self.enc_to_dec(encoded_tokens)

            # reapply decoder position embedding to unmasked tokens

            pos_masked_indices = masked_indices % (self.num_patches - 1)
            pos_unmasked_indices = unmasked_indices % (self.num_patches - 1)
            unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(pos_unmasked_indices)

            spectral_masked_indices = masked_indices // (self.num_patches - 1)
            spectral_unmasked_indices = unmasked_indices // (self.num_patches - 1)

            # Adapt the spectral indices to spectral band ids
            spectral_masked_ids = torch.zeros_like(spectral_masked_indices)
            for i in range(spectral_masked_indices.shape[0]):
                for j in range(spectral_masked_indices.shape[1]):
                    spectral_masked_ids[i, j] = token_channel_dict[spectral_masked_indices[i, j].item()]

            spectral_unmasked_ids = torch.zeros_like(spectral_unmasked_indices)
            for i in range(spectral_unmasked_indices.shape[0]):
                for j in range(spectral_unmasked_indices.shape[1]):
                    spectral_unmasked_ids[i, j] = token_channel_dict[spectral_unmasked_indices[i, j].item()]

            unmasked_decoder_tokens = unmasked_decoder_tokens + self.spectral_embedding(spectral_unmasked_ids)

            # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

            mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
            mask_tokens = mask_tokens + self.decoder_pos_emb(pos_masked_indices)
            mask_tokens = mask_tokens + self.spectral_embedding(spectral_masked_ids)
            # concat the masked tokens to the decoder tokens and attend with decoder

            decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
            decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
            decoder_tokens[batch_range, masked_indices] = mask_tokens
            decoded_tokens = self.decoder(decoder_tokens)

            # splice out the mask tokens and project to pixel values

            mask_tokens = decoded_tokens[batch_range, masked_indices]
            if self.configs["single_embedding_layer"]:
                pred_pixel_values = self.to_pixels(mask_tokens)
            else:
                # Project to_pixel for each modality of the decoded tokens. The modality is determined by the spectral_masked_indices
                # Get unique spectral masked ids
                unique_spectral_masked_ids = torch.unique(spectral_masked_ids)
                tokens_per_id = {}
                projected = torch.zeros(
                    size=(mask_tokens.shape[0], mask_tokens.shape[1], self.pixel_values_per_patch), device=device
                )
                for i in range(len(unique_spectral_masked_ids)):
                    self.to_pixels[unique_spectral_masked_ids[i].item()].to(device)

                    tokens_per_id[unique_spectral_masked_ids[i].item()] = mask_tokens[
                        spectral_masked_ids == unique_spectral_masked_ids[i]
                    ]
                    tmp_tokens = tokens_per_id[unique_spectral_masked_ids[i].item()].to(projected.dtype)
                    tmp_pixel_space = self.to_pixels[unique_spectral_masked_ids[i].item()](tmp_tokens)
                    projected = projected.to(tmp_pixel_space.dtype)
                    projected[spectral_masked_ids == unique_spectral_masked_ids[i]] = tmp_pixel_space

                pred_pixel_values = projected

            # calculate reconstruction loss

            recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
            return out, recon_loss
        else:
            return out, None
