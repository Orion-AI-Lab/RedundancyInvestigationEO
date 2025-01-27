import torch
import timm
from timm.layers import PatchEmbed
from timm.layers.format import Format, nchw_to
from torch import nn
import torch.nn.functional as F
import einops
import wandb
import MaskAttentionViT as MaskAttentionViT

class RViT(nn.Module):
    def __init__(self, num_classes, backbone, image_size, num_channels, pretrained=True, mask_threshold=0.9,mask_in_token_space=True, implementation="max_len", pad_value=0.0, retain_percent=0,attention_mask=False):
        """
        Initialize the RViT model.

        Args:
            num_classes: The number of classes in the dataset.
            backbone: The name of the backbone model to use.
            image_size: The size of the input images.
            num_channels: The number of channels in the input images.
            pretrained: Whether to use a pretrained model.
            mask_threshold: The threshold for masking similar tokens.
            mask_in_token_space: Whether to mask similar tokens in the token space or in the image space.
            pad_value: The value to use for padding masked tokens.
            attention_mask: Whether to use mask attention for the padded tokens
        """
        super(RViT, self).__init__()
        self.mask_in_token_space = mask_in_token_space
        self.mask_threshold = mask_threshold
        self.pad_value = pad_value
        self.retain_percent = retain_percent
        self.attention_mask = attention_mask
        self.implementation = implementation
        
        if not self.attention_mask:
            self.model = timm.create_model(backbone,in_chans = num_channels, num_classes=num_classes, img_size=image_size, dynamic_img_pad=True, dynamic_img_size=True,pretrained=pretrained)
        else:
            if self.implementation in ["median", "topx%"]:
                raise ValueError("Invalid arguments. "
                                "Change the attention_mask to False or the implementation to max_len")
            self.model = MaskAttentionViT._create_vision_transformer(backbone, img_size=image_size, num_classes=num_classes, in_chans=num_channels, pretrained=pretrained)
         # Ensure padding implementation is provided
            
        if self.implementation not in ["median", "max_len", "topx%"]:
            raise ValueError("Invalid or missing `implementation`. "
                                "Choose from: [median, max_len, topx%]")
        
        if self.implementation == "topx%" and self.retain_percent <= 0:
            raise ValueError("Value of retain_percent must be greater than 0 and lower than 100 for topx% implementation.")
        

    def calc_mask(self, x,verbose=False):
        #Mask similar tokens
        out = x[:,1:].mean(dim=0)

        #Calculate cosine similarity between all pairs of tokens
        norms_torch = torch.norm(out, dim=1, keepdim=True)
        normalized_tensor_torch = out / norms_torch
        cos_sim = torch.mm(normalized_tensor_torch, normalized_tensor_torch.T)
        keep_mask = torch.ones(cos_sim.shape[0], dtype=torch.bool)
        # Iterate over each pair of tokens
        for i in range(cos_sim.shape[0]):
            for j in range(i + 1, cos_sim.shape[1]):
                if i != j:
                    if cos_sim[i, j] > self.mask_threshold:
                        # If both tokens are similar, mark one for removal (let's keep the first one)
                        keep_mask[j] = False
        keep_mask =  torch.cat((torch.tensor([True]),keep_mask))
        x = x[:,keep_mask] 
        if verbose:
            print("Masked: ", keep_mask.sum())
            print("Masked ratio: ",(keep_mask.sum()-1)/keep_mask.shape[0])
            print(x.shape)
        if wandb.run:
            wandb.log({"mask_ratio":(keep_mask.sum()-1)/keep_mask.shape[0]})
        return x

    def batch_cos_similarity(self, x):
        """
        Calculate the cosine similarity between all pairs of tokens in a batch of samples.

        Args:
            x: A tensor of shape (batch_size, num_tokens, token_dim) representing the tokens.

        Returns:
            A tensor of shape (batch_size, num_tokens, num_tokens) containing the cosine similarity
            between all pairs of tokens in each sample of the batch.
        """
        norms_torch = torch.norm(x, dim=2, keepdim=True)
        normalized_tensor_torch = x / norms_torch
        cos_sim = torch.bmm(normalized_tensor_torch, normalized_tensor_torch.transpose(1, 2))
        return cos_sim
    
    def get_mask(self, cos_sim):
        """ 
        Get the mask of tokens to keep based on the cosine similarity between them.
        Args:
            cos_sim: A tensor of shape (batch,num_tokens, num_tokens) containing the cosine similarity
            between all pairs of tokens in a single sample.
        Returns:
            A tensor of shape (batch,num_tokens) containing a boolean mask indicating which tokens to keep per sample.
        """
        
        keep_mask = torch.ones(cos_sim.shape[0],cos_sim.shape[1]+1, dtype=torch.bool,device=cos_sim.device)
        # Iterate over each pair of tokens for the batch vectorised
        for b in range(cos_sim.shape[0]):
            for i in range(cos_sim.shape[1]):
                for j in range(i + 1, cos_sim.shape[2]):
                    if i != j:
                        if cos_sim[b,i, j] > self.mask_threshold:
                            # If both tokens are similar, mark one for removal (let's keep the first one)
                            keep_mask[b,j+1] = False
                            
        return keep_mask
    
    
    def get_mask_vectorised(self, cos_sim):
        """
        Get the mask of tokens to keep based on the cosine similarity between them.
        Args:
            cos_sim: A tensor of shape (batch,num_tokens, num_tokens) containing the cosine similarity
            between all pairs of tokens in a single sample.
        Returns:
            A tensor of shape (batch,num_tokens) containing a boolean mask indicating which tokens to keep per sample.
        """
        
        # Initialize the keep mask (batch, num_tokens+1) with True (keep all tokens initially)
        batch_size, num_tokens, _ = cos_sim.shape
        keep_mask = torch.ones((batch_size, num_tokens + 1), dtype=torch.bool, device=cos_sim.device)
        
        # Create a mask for cosine similarity above the threshold
        sim_mask = cos_sim > self.mask_threshold  # (batch, num_tokens, num_tokens)

        # Create an upper triangular mask to ignore duplicate pairs and self-pairs (i, i)
        triu_mask = torch.triu(torch.ones_like(sim_mask), diagonal=1).bool()  # (batch, num_tokens, num_tokens)

        # Combine the two masks (only consider the upper triangle and cosine similarities above the threshold)
        combined_mask = sim_mask & triu_mask  # (batch, num_tokens, num_tokens)

        # Find which tokens should be marked False (don't keep). We mark the second token (j)
        # Reduce over the third axis (num_tokens) to see if any token j should be masked
        mark_false = combined_mask.any(dim=1)  # (batch, num_tokens)
        
        # Update the keep_mask, skipping the first token (keep_mask[:, 0] is ignored)
        keep_mask[:, 1:] &= ~mark_false

        return keep_mask

    def calc_mask_vectorised(self, x,verbose=False):
        #Same as calc_mask but operates on the whole batch without averaging
        out = x[:,1:]     
        total_patches = out.shape[1]   
        cos_sim = self.batch_cos_similarity(out)
        
        keep_mask = self.get_mask_vectorised(cos_sim)
        print("Checkpoint1", keep_mask[1,:].tolist())
        #Keep mask per row sum
        retained_tokens = [x[i, keep_mask[i]] for i in range(x.shape[0])]

        #Get median length
        #median_len = torch.median(torch.tensor([tokens.shape[0] for tokens in retained_tokens]))

        # Count the number of tokens retained for each sample
        retained_counts = keep_mask.sum(dim=1).tolist()
        print("Retained tokens per sample:", retained_counts)

        
        # Get max length (instead of median)
        max_len = max(tokens.shape[0] for tokens in retained_tokens)
        max_len = min(max_len, total_patches)
        #Calc topk indices
        topk_indices = self.get_topk_indices(cos_sim,max_len)
        if self.model.global_pool=='token':
            #Increment indices by 1 to account for the CLS token
            topk_indices += 1
        
        batch_indices = torch.arange(x.shape[0], device=x.device)[:, None]
        x = x[batch_indices,topk_indices]

        # Pad to max_len with zeros (or another suitable value)
        #pad_value = 0  # Change if necessary
        padded_x = torch.full((x.shape[0], max_len, x.shape[2]), self.pad_value, device=x.device)

        for i in range(x.shape[0]):
            padded_x[i, :x[i].shape[0]] = x[i]  

        #masked_ratio_calculation
        # Count how many patches are masked per sample (excluding CLS token)
        masked_patches_per_sample = (~keep_mask[:, 1:]).sum(dim=1)
        unmasked_patches_per_sample = (keep_mask[:, 1:]).sum(dim=1)

        # Sum patches across the whole batch
        masked_patches_in_batch = masked_patches_per_sample.sum()
        unmasked_patches_in_batch = unmasked_patches_per_sample.sum() #chech if it is pushed

        # Compute masked ratio
        masked_ratio = masked_patches_in_batch / (x.shape[0] * (x.shape[1] - 1))  # Normalize by batch size
        unmasked_ratio = unmasked_patches_in_batch / (x.shape[0] * (x.shape[1] - 1))

        if verbose:
            print("X shape: ", padded_x.shape)
            print("Keep mask shape: ", keep_mask.shape)
            print("Masked in batch: ", masked_patches_in_batch)
            print("Masked ratio: ", masked_ratio)
            print(x.shape)
        if wandb.run:
            wandb.log({"unmasked_patches_ratio":unmasked_ratio})
        return x


    def forward_mask_in_token_space(self, x,verbose=False):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        #Mask per batch
        if self.training:
            x = self.calc_mask_vectorised(x,False)

        #TODO Mask per sample and pad to max length (?)
        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        return x
    

    def get_topk_indices(self,cos_sim,k):
        """ From a cosine similarity matrix, get the top k indices indicating the most informative tokens for each sample.

        Args:
            cos_sim: A tensor of shape (batch,num_tokens, num_tokens) containing the cosine similarity
            k: The number of tokens to keep.
        Returns:
            A tensor of shape (batch,k) containing the indices of the top k tokens for each sample.
        """
        mask = ~torch.eye(cos_sim.shape[1], dtype=bool, device=cos_sim.device)
        masked_similarities = cos_sim * mask.unsqueeze(0)
        mean_similarities = masked_similarities.sum(dim=2) / (cos_sim.shape[1] - 1)
        tok_k_indices = torch.topk(-mean_similarities, k, dim=1).indices
        return tok_k_indices
    
    def forward_mask_in_image_space(self, x, verbose=False):
        """
        Mask patches based on cosine similarity and retain the most important ones.
        Masked patches are replaced with padding.
        """
        if self.training and self.mask_threshold < 1.0 and self.retain_percent < 100:
            patch_size = self.model.patch_embed.patch_size[0]

            # Ensure input dimensions are multiples of patch size
            H, W = x.shape[2], x.shape[3]
            pad_h, pad_w = (patch_size - H % patch_size) % patch_size, (patch_size - W % patch_size) % patch_size
            x = F.pad(x, (0, pad_w, 0, pad_h))

            # Extract patches
            patches = einops.rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)

            # Compute cosine similarity
            cos_sim = self.batch_cos_similarity(patches)


        # Extract actual token embeddings
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)

        # Handle CLS token separately
        cls_token = None
        if self.model.global_pool == 'token':
            cls_token, x = x[:, :1, :], x[:, 1:]  # Separate CLS token

        total_patches = x.shape[1]  # Exclude CLS token count
        
        #3 Options: 1st-topx%, 2nd-median, 3rd-max_len
        if self.implementation == "topx%":
            # Get top-k indices (ensures fixed-size output)
            keep_indices = self.get_topk_indices(cos_sim, int(cos_sim.shape[1] * (self.retain_percent / 100)))
            if self.training and self.retain_percent < 100:
                # Ensure we select the same number of patches across all samples
                k = keep_indices.shape[1]  # Number of patches to retain
                # Gather selected patches
                batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(-1).expand(-1, k)
                x = x[batch_indices, keep_indices]  # Efficient batch gather

                # Re-add CLS token if necessary
                if self.model.global_pool == 'token':
                    cls_token = cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat([cls_token, x], dim=1)

                if verbose:
                    print(f"Cos sim shape: {cos_sim.shape}, Selected indices: {keep_indices.shape}, Output shape: {x.shape}")

                if wandb.run:
                    wandb.log({"unmasked_patches_ratio": k / total_patches})
                print("topx%", k, self.retain_percent)

        elif self.implementation == "max_len":
            # Get binary mask (True = keep, False = mask)
            keep_mask = self.get_mask_vectorised(cos_sim)
            if self.training and self.mask_threshold < 1.0:
                keep_mask = keep_mask[:, :total_patches]  # Ensure matching shape

                # **Vectorized Retained Tokens Selection**
                lens = keep_mask.sum(dim=1)  # Get number of unmasked patches per image
                #print("lens:", lens)
                max_len = min(lens.max().item(), total_patches)

                #padded tensor efficient creation
                padded_x = torch.full((x.shape[0], max_len, x.shape[2]), self.pad_value, device=x.device, dtype=x.dtype)

                #Use Masking to Copy Retained Tokens
                mask = torch.arange(total_patches, device=x.device).unsqueeze(0) < lens.unsqueeze(1)
                padded_x[mask[:, :max_len]] = x[keep_mask]  # Efficiently copy retained tokens

                x = padded_x  # Update x with masked patches

                if self.attention_mask:
                    token_attention_mask = torch.zeros((x.shape[0], max_len), dtype=torch.bool, device=x.device)
                    token_attention_mask[~mask[:, :max_len]] = True  # True for masked-out tokens

                # Re-add CLS token if necessary
                if self.model.global_pool == 'token':
                    cls_token = cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat([cls_token, x], dim=1)

                    if self.attention_mask:
                        cls_mask = torch.zeros((x.shape[0], 1), device=x.device, dtype=torch.bool)
                        token_attention_mask = torch.cat((cls_mask, token_attention_mask), dim=1)

                if verbose:
                    print(f"Cos sim shape: {cos_sim.shape}, Mask shape: {keep_mask.shape}, Unmasked shape: {x.shape}, Max len: {max_len}")

                if wandb.run:
                    wandb.log({"unmasked_patches_ratio": max_len / total_patches})
                print("max_len", max_len)

        else: #median
            # Get binary mask (True = keep, False = mask)
            keep_mask = self.get_mask_vectorised(cos_sim)
            if self.training and self.mask_threshold<1.0:   
                keep_mask = keep_mask[:, :total_patches]  # Ensure matching shape     
            #Mask based on similarity
                retained_tokens = [x[i, keep_mask[i]] for i in range(x.shape[0])]
                
                #Get median length
                lengths = torch.tensor([tokens.shape[0] for tokens in retained_tokens], device=x.device)
                median_len = int(torch.median(lengths).item())
                
                #Calc topk indices
                topk_indices = self.get_topk_indices(cos_sim,median_len)
               # Re-add CLS token if necessary
                if self.model.global_pool == 'token':
                    cls_token = cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat([cls_token, x], dim=1)
                
                batch_indices = torch.arange(x.shape[0], device=x.device)[:, None]
                x = x[batch_indices,topk_indices]
                

                if verbose:
                    print(f"Cos sim shape: {cos_sim.shape}, Mask shape: {keep_mask.shape}, Unmasked shape: {x.shape}, Max len: {max_len}")

                if wandb.run:
                    wandb.log({"unmasked_patches_ratio": median_len / total_patches})
                print("median", median_len / total_patches)
                print("DEBUG: Retained token lengths:", [tokens.shape[0] for tokens in retained_tokens])
                print("DEBUG: Median length before casting:", median_len)
                print("DEBUG: Cosine similarity shape:", cos_sim.shape)
                print(f"batch_indices shape: {batch_indices.shape}, topk_indices shape: {topk_indices.shape}")
                print("DEBUG: x shape:", x.shape)


        # Forward through the model
        x = self.model.norm_pre(x)

        if self.attention_mask:
            token_attention_mask = token_attention_mask[:, None, None, :]  # Shape it for attention layers
            for module in self.model.blocks:
                x = module(x, token_attention_mask)
        else:
            x = self.model.blocks(x)

        x = self.model.norm(x)

        return x


    
    def forward(self, x):
        if self.mask_in_token_space:
            x = self.forward_mask_in_token_space(x,verbose=False)
        else:
            x = self.forward_mask_in_image_space(x,verbose=False)

        return self.model.forward_head(x)


if __name__ == '__main__':
    model = RViT(num_classes=14, backbone='vit_base_patch16_224', image_size=128, num_channels=14, mask_threshold=0.9, mask_in_token_space=False, implementation="max_len", pad_value = 0.0, retain_percent=0, attention_mask=False)
    
    def create_rand():
        # Step 1: Create a random 16x16 block
        block = torch.randn(14, 16, 16)

        # Step 2: Concatenate 8 blocks horizontally (along the width)
        row = torch.cat([block] * 8, dim=2)  # Shape: (14, 16, 128)

        # Step 3: Concatenate 8 such rows vertically (along the height)
        tensor_8x8 = torch.cat([row] * 8, dim=1)  # Shape: (14, 128, 128)
        x = tensor_8x8.unsqueeze(0)
        return x
    x = create_rand()
    for i in range(10):
        x = torch.cat((x,create_rand()),dim=0)
        
    x = torch.cat((x,torch.randn(1,14,128,128)),dim=0)
    x = torch.cat((x,torch.randn(1,14,128,128)),dim=0)
    model.train()
    out = model(x)
    print(out.shape)