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

    def batch_cos_similarity(self,x):
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
        cos_sim = torch.bmm(normalized_tensor_torch, normalized_tensor_torch.transpose(1,2))
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
        
        #Keep mask per row sum
        retained_tokens = [x[i, keep_mask[i]] for i in range(x.shape[0])]
                
        #Get median length
        median_len = torch.median(torch.tensor([tokens.shape[0] for tokens in retained_tokens]))
        
        #Calc topk indices
        topk_indices = self.get_topk_indices(cos_sim,median_len)
        if self.model.global_pool=='token':
            #Increment indices by 1 to account for the CLS token
            topk_indices += 1
        
        batch_indices = torch.arange(x.shape[0], device=x.device)[:, None]
        x = x[batch_indices,topk_indices]
        
        if verbose:
            print("X shape: ",x.shape)
            print("Keep mask shape: ",keep_mask.shape)
            print("Masked: ", keep_mask.sum())
            print("Masked ratio: ",median_len/total_patches)
            print(x.shape)
        if wandb.run:
            wandb.log({"unmasked_patches_ratio":median_len/total_patches})
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
    
    def forward_mask_in_image_space(self, x,verbose=False):
        #Mask per batch
        if self.training and self.mask_threshold<1.0:
            patch_size = self.model.patch_embed.patch_size[0]

            H = x.shape[2]
            W = x.shape[3]
            pad_h = (patch_size - H % patch_size) % patch_size
            pad_w = (patch_size - W % patch_size) % patch_size
            tmp_x = F.pad(x, (0, pad_w, 0, pad_h))
            patches = einops.rearrange(tmp_x,"b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)

            cos_sim = self.batch_cos_similarity(patches)
            #keep_mask = self.get_mask(cos_sim)
            keep_mask = self.get_mask_vectorised(cos_sim)

        #Get actual tokens
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        total_patches = x.shape[1] -1

        #Mask per batch
        if self.training and self.mask_threshold<1.0:        
            #Mask based on similarity
            retained_tokens = [x[i, keep_mask[i]] for i in range(x.shape[0])]
            
            #Get median length
            median_len = torch.median(torch.tensor([tokens.shape[0] for tokens in retained_tokens]))
            
            #Calc topk indices
            topk_indices = self.get_topk_indices(cos_sim,median_len)
            if self.model.global_pool=='token':
                #Increment indices by 1 to account for the CLS token
                topk_indices += 1
            
            batch_indices = torch.arange(x.shape[0], device=x.device)[:, None]
            x = x[batch_indices,topk_indices]
            if verbose:
                print("Cos sim shape: ",cos_sim.shape)
                print("Indices shape: ",topk_indices.shape)
                print("Unmasked shape: ",x.shape)
                print("Median masked shape: ",x[batch_indices,topk_indices].shape)            
                print("Median len: ",median_len)            
        
            if wandb.run:
                wandb.log({"unmasked_patches_ratio":median_len/total_patches})
            
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
    model = RViT(num_classes=14, backbone='vit_base_patch16_224', image_size=120, num_channels=14,mask_threshold=0.8,mask_in_token_space=False)
    x = torch.randn(64, 14, 60, 60)
    
    x = torch.cat((x,x),dim=2)
    print(x.shape)
    
    x = torch.cat((x,x),dim=3)
    print(x.shape)
    model.train()
    out = model(x)
    print(out.shape)