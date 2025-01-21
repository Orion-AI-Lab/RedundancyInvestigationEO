import torch
import timm
from timm.layers import PatchEmbed
from timm.layers.format import Format, nchw_to
from torch import nn
import torch.nn.functional as F
import einops
import wandb

class RViT(nn.Module):
    def __init__(self, num_classes, backbone, image_size, num_channels, pretrained=True, mask_threshold=0.9,mask_in_token_space=True):
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
        """
        super(RViT, self).__init__()
        self.mask_in_token_space = mask_in_token_space
        self.mask_threshold = mask_threshold
        self.model = timm.create_model(backbone,in_chans = num_channels, num_classes=num_classes, img_size=image_size, dynamic_img_pad=True, dynamic_img_size=True,pretrained=pretrained)
        #print(self.model)

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
        print("Cosine Siilarity:", cos_sim[1, 1, :])
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
        counter = keep_mask.sum(dim=1).tolist()
        #print("Keep_mask:", counter)
        
        # Create a mask for cosine similarity above the threshold
        sim_mask = cos_sim > self.mask_threshold  # (batch, num_tokens, num_tokens)

        # Create an upper triangular mask to ignore duplicate pairs and self-pairs (i, i)
        triu_mask = torch.triu(torch.ones_like(sim_mask), diagonal=1).bool()  # (batch, num_tokens, num_tokens)
       # print(triu_mask[1,1])
        # Combine the two masks (only consider the upper triangle and cosine similarities above the threshold)
        combined_mask = sim_mask & triu_mask  # (batch, num_tokens, num_tokens)
        #print(combined_mask)
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
        pad_value = 0  # Change if necessary
        padded_x = torch.full((x.shape[0], max_len, x.shape[2]), pad_value, device=x.device)

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
            #print(padded_x.shape)
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
        print(tok_k_indices[0, :])
        return tok_k_indices
    
    def forward_mask_in_image_space(self, x, verbose=False):
        """
        Mask patches based on cosine similarity and retain the most important ones.
        Masked patches are replaced with padding.
        """
        if self.training and self.mask_threshold < 1.0:
            patch_size = self.model.patch_embed.patch_size[0]
            
            # Ensure input dimensions are multiples of patch size
            H, W = x.shape[2], x.shape[3]
            pad_h, pad_w = (patch_size - H % patch_size) % patch_size, (patch_size - W % patch_size) % patch_size
            tmp_x = F.pad(x, (0, pad_w, 0, pad_h))
            
            # Extract patches
            patches = einops.rearrange(tmp_x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)
            
            # Compute cosine similarity
            cos_sim = self.batch_cos_similarity(patches)

            # Get binary mask (True = keep, False = mask)
            keep_mask = self.get_mask_vectorised(cos_sim)

        # Extract actual token embeddings
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)

        # Handle CLS token separately
        cls_token = None
        if self.model.global_pool == 'token':
            cls_token, x = x[:, :1, :], x[:, 1:]  # Separate CLS token

        total_patches = x.shape[1]  # No CLS token in count

        if self.training and self.mask_threshold < 1.0:
            # **Fix keep_mask Shape Issue**
            keep_mask = keep_mask[:, :total_patches]  # Ensure it matches x.shape[1]

            # Keep only unmasked patches
            retained_tokens = [x[i, keep_mask[i]] for i in range(x.shape[0])]

            # Get number of retained tokens per image
            lens = torch.tensor([tokens.shape[0] for tokens in retained_tokens], dtype=torch.int32)

            # Get max sequence length (ensuring it does not exceed total_patches)
            max_len = min(max(lens), total_patches)
            print("Max_len:", max_len)

            # Create a new tensor with padding
            pad_value = 0  # Can be changed if needed
            padded_x = torch.full((x.shape[0], max_len, x.shape[2]), pad_value, device=x.device)

            # Copy retained tokens into padded_x
            for img in range(x.shape[0]):
                num_tokens = retained_tokens[img].shape[0]
                padded_x[img, :num_tokens] = retained_tokens[img]

            x = padded_x

            # Add CLS token back if needed
            if self.model.global_pool == 'token':
                cls_token = cls_token.expand(x.shape[0], -1, -1)  # Ensure CLS has correct batch size
                x = torch.cat([cls_token, x], dim=1)

            if verbose:
                print("Cos sim shape: ", cos_sim.shape)
                print("Mask shape: ", keep_mask.shape)
                print("Unmasked shape: ", x.shape)
                print("Max masked shape: ", x.shape)
                print("Max len: ", max_len)

            if wandb.run:
                wandb.log({"unmasked_patches_ratio": max_len / total_patches})

        # Forward through the model
        x = self.model.norm_pre(x)
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
    model = RViT(num_classes=14, backbone='vit_base_patch16_224', image_size=128, num_channels=14,mask_threshold=0.9,mask_in_token_space=False)
    #x = torch.randn(64, 14, 64, 64)
    #x = torch.ones(64, 14, 128, 128)
    #x = torch.cat((x,x),dim=2)
    x = torch.randn(64,14, 128, 128)
    #print(x.shape)
    #x = torch.cat((x,x),dim=3)
    print(x.shape)
    model.train()
    out = model(x)
    print(out.shape)