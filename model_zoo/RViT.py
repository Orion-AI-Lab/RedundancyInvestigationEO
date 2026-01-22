import torch
import timm
from timm.layers import PatchEmbed
#from timm.layers.format import Format, nchw_to
from torch import nn
import torch.nn.functional as F
import einops
import wandb
import hashlib
import numpy as np

# Import torch._dynamo for skipping compilation on functions that use Python strings
try:
    import torch._dynamo as dynamo
    skip = dynamo.skip
    # Mark dynamic for parameters that change but shouldn't cause recompilation
    try:
        mark_dynamic = dynamo.mark_dynamic
    except AttributeError:
        mark_dynamic = None
except (ImportError, AttributeError):
    # Fallback if torch._dynamo is not available
    def skip(fn):
        return fn
    mark_dynamic = None

class RViT(nn.Module):
    def __init__(self, num_classes, backbone, image_size, num_channels, pretrained=True, mask_threshold=0.9,mask_in_token_space=True, implementation="max_len", pad_value=0.0, retain_percent=0,attention_mask=False, hierarchical_features=False):
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
        self.backbone = backbone
        self.hierarchical_features = hierarchical_features
        self.plot_img = True
        self.plot_cnt = 0

        if not self.attention_mask:
            self.model = timm.create_model(backbone,in_chans = num_channels, num_classes=num_classes, img_size=image_size, dynamic_img_pad=True, dynamic_img_size=True,pretrained=pretrained)
            #self.model.patch_embed.set_input_size(image_size, patch_size=patch_size)
        else:
            from model_zoo import MaskAttentionViT as MaskAttentionViT

            if self.implementation in ["median", "topx%", "vit", "random_topx%"]:
                raise ValueError("Invalid arguments. "
                                "Change the attention_mask to False or the implementation to max_len")

            self.model = timm.create_model(backbone, img_size=image_size, num_classes=num_classes, dynamic_img_pad=True, dynamic_img_size=True, in_chans=num_channels, pretrained=pretrained)
            #self.model.patch_embed.set_input_size(image_size, patch_size)
            

        #Ensure padding implementation is provided
        self.num_blocks = len(self.model.blocks) 
        if self.hierarchical_features:
            # Dynamically divide blocks into stages
            self.stage1_blocks = self.model.blocks[:self.num_blocks // 4]
            self.stage2_blocks = self.model.blocks[self.num_blocks // 4:self.num_blocks // 2]
            self.stage3_blocks = self.model.blocks[self.num_blocks // 2:3 * self.num_blocks // 4]
            self.stage4_blocks = self.model.blocks[3 * self.num_blocks // 4:]
            
        if hasattr(self.model, 'MaskAttentionViT'):
            print("Initiliazing MaskAttentionViT")
            if wandb.run:
                wandb.config.update({"MaskAttentionViT":True})
            if not self.attention_mask:
                raise ValueError("MaskAttentionViT requires attention_mask=True")
              
        if self.implementation not in ["median", "max_len", "topx%", "vit", "random_topx%"]:
            raise ValueError("Invalid or missing `implementation`. "
                                "Choose from: [median, max_len, topx%, vit, random_topx%]")
        
        if (self.implementation == "topx%" or self.implementation == "random_topx%") and (self.retain_percent <= 0 or self.retain_percent > 100):
            raise ValueError("Value of retain_percent must be greater than 0 and less than or equal to 100 for topx% implementation.")
        


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
        # print("Checkpoint1", keep_mask[1,:].tolist())  # Commented out for torch.compile compatibility
        #Keep mask per row sum
        retained_tokens = [x[i, keep_mask[i]] for i in range(x.shape[0])]

        #Get median length
        #median_len = torch.median(torch.tensor([tokens.shape[0] for tokens in retained_tokens]))

        # Count the number of tokens retained for each sample
        retained_counts = keep_mask.sum(dim=1).tolist()
        
        # Get max length (instead of median)
        max_len = max(tokens.shape[0] for tokens in retained_tokens)
        max_len = min(max_len, total_patches)
        #Calc topk indices
        topk_indices = self.get_topk_indices_redundancy_based(cos_sim,max_len)
        #topk_indices = self.get_topk_indices_mean_similarity(cos_sim,max_len)
        # Sort indices to ensure consistent ordering
        topk_indices, _ = torch.sort(topk_indices, dim=1)
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
            # print("X shape: ", padded_x.shape)  # Commented out for torch.compile compatibility
            # print("Keep mask shape: ", keep_mask.shape)  # Commented out for torch.compile compatibility
            # print("Masked in batch: ", masked_patches_in_batch)  # Commented out for torch.compile compatibility
            # print("Masked ratio: ", masked_ratio)  # Commented out for torch.compile compatibility
            # print(x.shape)  # Commented out for torch.compile compatibility
            pass
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
    

    def get_topk_indices_mean_similarity(self,cos_sim,k):
        """ From a cosine similarity matrix, get the top k indices indicating the most informative tokens for each sample.

        Args:
            cos_sim: A tensor of shape (batch,num_tokens, num_tokens) containing the cosine similarity
            k: The number of tokens to keep.
        Returns:
            A tensor of shape (batch,k) containing the indices of the top k tokens for each sample.
        """
        cos_sim = torch.abs(cos_sim)
        mask = ~torch.eye(cos_sim.shape[1], dtype=torch.bool, device=cos_sim.device)
        masked_similarities = cos_sim * mask.unsqueeze(0)
        mean_similarities = masked_similarities.sum(dim=2) / (cos_sim.shape[1] - 1)
        tok_k_indices = torch.topk(-mean_similarities, k, dim=1).indices
        # Sort indices to ensure consistent ordering across different selection approaches
        tok_k_indices, _ = torch.sort(tok_k_indices, dim=1)
        return tok_k_indices

    def get_topk_indices_redundancy_based_min(self,cos_sim, k):
        """Keep one patch from each high-similarity pair, without duplicates.
        Fully vectorized across batch. Returns tensor shape [batch, k] with original patch indices.
        """
        device = cos_sim.device
        batch_size = cos_sim.shape[0]
        num_patches = cos_sim.shape[1]
        eps = 1e-7

        cos_sim = torch.abs(cos_sim) + eps  # [B, n, n]

        if k >= num_patches:
            # Return all patches in order
            return torch.arange(num_patches, device=device).unsqueeze(0).expand(batch_size, -1)

        # keep only upper triangular (row < col)
        mask = torch.triu(torch.ones((num_patches, num_patches), dtype=torch.bool, device=device), diagonal=1)
        masked = cos_sim * mask.unsqueeze(0)  # [B, n, n]

        # Replace zero/unmasked entries with -inf so they are never selected by max/topk
        masked = masked.clone()
        masked[masked == 0] = float('inf')

        masked_flat = masked.view(batch_size, -1)  # [B, n*n]
        result = torch.full((batch_size, k), -1, dtype=torch.long, device=device)

        base = torch.arange(num_patches, device=device)  # [n]
        batch_idx = torch.arange(batch_size, device=device)

        for i in range(k):
            # pick the highest similarity pair per batch
            vals, idx = masked_flat.min(dim=1)  # [B], [B] (max avoids dealing with sign inversions)
            # if all remaining entries are -inf, idx will be some index but vals == -inf -> break early
            if torch.isneginf(vals).all():
                # no more available pairs -> fill remaining slots with -1 or any sentinel
                break

            row = idx // num_patches       # [B]
            col = idx % num_patches        # [B]

            # Decide which patch to keep from the pair (here: keep the 'row' element)
            chosen = row  # shape [B]
            result[:, i] = chosen

            # Now set to -inf all flattened entries that involve `chosen` patch
            # 1) Pairs where chosen is the row (chosen as first element): chosen * n + j for j=0..n-1
            rows_chosen = chosen.unsqueeze(1) * num_patches + base.unsqueeze(0)   # [B, n]
            masked_flat[batch_idx.unsqueeze(1), rows_chosen] = float('inf')

            # 2) Pairs where chosen is the column (chosen as second element): j * n + chosen for j=0..n-1
            cols_chosen = base.unsqueeze(0) * num_patches + chosen.unsqueeze(1)   # [B, n]
            masked_flat[batch_idx.unsqueeze(1), cols_chosen] = float('inf')
            # (The selected pair itself is included in the above and is set to -inf too.)

        return result


    def get_topk_indices_redundancy_based(self, cos_sim, k):
        """
        Vectorized across batch: repeatedly find the pair with maximum cosine similarity
        (upper-triangular pairs only), drop one element from that pair, do this
        (num_patches - k) times, then return the indices of the k remaining patches.

        Returns: Tensor of shape [batch, k] with original patch indices (ascending order).
        If fewer than k remain (very unlikely), remaining slots are filled with -1.
        """
        device = cos_sim.device
        batch_size = cos_sim.shape[0]
        num_patches = cos_sim.shape[1]
        #eps = 1e-7

        # trivial cases
        if k >= num_patches:
            return torch.arange(num_patches, device=device).unsqueeze(0).expand(batch_size, -1).clone()

        # absolute similarity so sign doesn't matter
        sim = torch.abs(cos_sim) #+ eps  # [B, n, n]
        # sim = ssim + eps  # [B, n, n]
        # upper triangular mask (row < col)
        tri_mask = torch.triu(torch.ones((num_patches, num_patches), dtype=torch.bool, device=device), diagonal=1)
        # invalid entries should be -inf so max ignores them
        sim = sim.clone()
        sim[:, ~tri_mask] = float("-inf")  # set diagonal and lower-tri to -inf

        # flattened view for efficient max across pairs
        sim_flat = sim.view(batch_size, -1)  # [B, n*n]

        # track dropped patches per batch
        dropped = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)

        base = torch.arange(num_patches, device=device)  # [n]
        drops = num_patches - k

        for _ in range(drops):
            vals, idx = sim_flat.max(dim=1)  # [B], [B]
            valid = ~torch.isneginf(vals)    # batches that have a valid pair to drop from
            if not valid.any():
                break

            # decode row/col from flat index
            row = idx // num_patches   # [B]
            col = idx % num_patches    # [B]

            # For invalid batches avoid indexing issues: set their chosen drop to 0 (won't be used)
            drop = col.clone()
            drop[~valid] = 0  # safe sentinel for invalid batches

            # mark dropped patches (vectorized)
            # create batch indices for boolean indexing
            vb = valid.nonzero(as_tuple=True)[0]
            if vb.numel() > 0:
                # mark as dropped for valid batches
                dropped[vb, drop[vb]] = True

                # set all flattened entries that involve the dropped patch to -inf (for those valid batches)
                # pairs where dropped is row: drop * n + j  (for j=0..n-1)
                rows_dropped = drop.unsqueeze(1) * num_patches + base.unsqueeze(0)   # [B, n]
                # pairs where dropped is col: j * n + drop
                cols_dropped = base.unsqueeze(0) * num_patches + drop.unsqueeze(1)   # [B, n]

                # apply only to valid batches (vectorized)
                sim_flat[vb.unsqueeze(1), rows_dropped[vb]] = float("-inf")
                sim_flat[vb.unsqueeze(1), cols_dropped[vb]] = float("-inf")

                # rebuild sim view and also ensure any pair involving any already-dropped patch is -inf
                sim = sim_flat.view(batch_size, num_patches, num_patches)

                # alive mask and pair-wise alive
                alive = ~dropped                       # [B, n]
                alive_i = alive.unsqueeze(2)           # [B, n, 1]
                alive_j = alive.unsqueeze(1)           # [B, 1, n]
                pair_alive = alive_i & alive_j & tri_mask.unsqueeze(0)  # [B, n, n]

                sim[~pair_alive] = float("-inf")
                sim_flat = sim.view(batch_size, -1)

        # After all drops: keep patches where dropped == False -> alive
        alive = ~dropped  # [B, n]

        # We want the indices of alive patches (sorted ascending) per batch, up to k entries.
        # Vectorized trick: create index matrix and set indices of dead patches to a sentinel (n)
        idxs = torch.arange(num_patches, device=device).unsqueeze(0).expand(batch_size, -1).clone()  # [B, n]
        idxs_masked = idxs.clone()
        idxs_masked[~alive] = num_patches  # sentinel that sorts to the end

        # sort along dim=1 and take first k (these are the alive indices in ascending order)
        sorted_idxs = torch.sort(idxs_masked, dim=1)[0]  # [B, n]
        topk_keep = sorted_idxs[:, :k]                  # [B, k]

        # replace sentinel (num_patches) with -1 for padding if any
        topk_keep[topk_keep == num_patches] = -1

        return topk_keep


    def get_topk_indices_drop_more_redundant(self, cos_sim, k):
        """
        Greedy drop strategy:
        - cos_sim: [B, n, n] pairwise similarity (can be asymmetric but usually symmetric)
        - k: number of patches to keep
        Algorithm (vectorized across batch):
        - work with abs(cos_sim)
        - consider only upper-triangular pairs (row < col)
        - repeat (n - k) times:
            * find max-similarity pair (row, col) per batch
            * compute per-patch mean similarity (over currently-alive patches)
            * drop, for each batch, the patch from {row, col} that has the **higher** mean similarity
                (i.e., more redundant)
            * mask out all pairs that involve the dropped patch
        - return indices of alive (not-dropped) patches per batch, sorted ascending, shape [B, k]
            padded with -1 if fewer than k remain (shouldn't normally happen)
        """
        device = cos_sim.device
        B = cos_sim.shape[0]
        n = cos_sim.shape[1]

        if k >= n:
            return torch.arange(n, device=device).unsqueeze(0).expand(B, -1).clone()

        # Work with absolute similarity so sign doesn't matter
        sim = torch.abs(cos_sim).clone()  # [B, n, n]

        # Upper triangular mask
        tri_mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)  # [n, n]
        sim[:, ~tri_mask] = float("-inf")  # set diag and lower-tri to -inf so max ignores them

        sim_flat = sim.view(B, -1)  # [B, n*n]

        dropped = torch.zeros((B, n), dtype=torch.bool, device=device)  # dropped mask per batch

        base = torch.arange(n, device=device)  # [n]
        drops = n - k
        batch_idx = torch.arange(B, device=device)

        for _ in range(drops):
            vals, idx = sim_flat.max(dim=1)  # [B], [B]
            valid = ~torch.isneginf(vals)     # batches that still have a valid pair

            if not valid.any():
                break

            # decode row/col of chosen max pair
            row = idx // n
            col = idx % n

            # For invalid batches, set safe sentinel (we won't use them)
            row[~valid] = 0
            col[~valid] = 0

            # compute mean similarity per patch using the current sim and alive mask.
            # We want the mean over other alive patches (ignore -inf entries).
            # Rebuild sim as [B,n,n] from sim_flat (it already has -inf for invalid pairs).
            sim = sim_flat.view(B, n, n)

            # alive mask (excluding already dropped patches)
            alive = ~dropped  # [B, n]
            alive_i = alive.unsqueeze(2)   # [B, n, 1]
            alive_j = alive.unsqueeze(1)   # [B, 1, n]
            pair_alive = alive_i & alive_j & tri_mask.unsqueeze(0)  # [B, n, n] only upper-tri alive pairs

            # For mean similarity per patch we want to consider all alive partners (both sides).
            # Build a symmetric view of pairwise alive similarities:
            # We'll create a matrix M where M[b,i,j] is sim[b,i,j] if pair (i,j) is alive (upper-tri),
            # and we mirror it to also give sim[b,j,i] = sim[b,i,j], so each patch's mean uses
            # its alive connections.
            # Extract upper-tri alive sim (with -inf for others)
            upper_alive_sim = sim.clone()
            upper_alive_sim[~pair_alive] = float("-inf")

            # Mirror to get symmetric alive similarity matrix (diagonal remains -inf)
            sym_alive_sim = upper_alive_sim + upper_alive_sim.transpose(1, 2)  # [B,n,n]
            # Now for each patch i, the alive similarities to others are sym_alive_sim[b,i,:]
            # Convert -inf to 0 for sum and count alive neighbors
            mask_alive_neighbors = (sym_alive_sim != float("-inf"))
            # sum and count
            sum_sim = torch.where(mask_alive_neighbors, sym_alive_sim, torch.zeros_like(sym_alive_sim)).sum(dim=2)  # [B, n]
            cnt_neighbors = mask_alive_neighbors.sum(dim=2).clamp(min=1)  # avoid div by zero, [B,n]
            mean_sim = sum_sim / cnt_neighbors  # [B, n] mean similarity to alive neighbors (0 if no neighbors)

            # Choose which to drop: between row and col, drop the patch with higher mean_sim (more redundant).
            # For batches that are invalid (no valid pair), we ignore by keeping a sentinel later.
            drop = torch.where(mean_sim[batch_idx, row] > mean_sim[batch_idx, col], row, col)
            # Tie-breaking: if equal mean_sim, drop the higher-index patch to be deterministic
            equal_mask = (mean_sim[batch_idx, row] == mean_sim[batch_idx, col]) & valid
            if equal_mask.any():
                drop[equal_mask] = torch.max(row[equal_mask], col[equal_mask])

            # apply to only valid batches
            vb = valid.nonzero(as_tuple=True)[0]
            if vb.numel() == 0:
                break

            dropped[vb, drop[vb]] = True

            # mask out all entries involving dropped patch (set to -inf in sim_flat)
            # pairs where dropped is row: drop * n + j  (for j=0..n-1)
            rows_dropped = drop.unsqueeze(1) * n + base.unsqueeze(0)   # [B, n]
            # pairs where dropped is col: j * n + drop
            cols_dropped = base.unsqueeze(0) * n + drop.unsqueeze(1)   # [B, n]

            sim_flat[vb.unsqueeze(1), rows_dropped[vb]] = float("-inf")
            sim_flat[vb.unsqueeze(1), cols_dropped[vb]] = float("-inf")

            # rebuild sim_flat (already updated) and continue
            # additionally ensure any pair involving an already-dropped patch is -inf
            # (this will be enforced by pair_alive recomputation in next loop iteration)

        # After drops: alive = ~dropped
        alive = ~dropped  # [B, n]

        # build indices and mask dead ones to sentinel n so they sort to the end
        idxs = torch.arange(n, device=device).unsqueeze(0).expand(B, -1).clone()  # [B, n]
        idxs_masked = idxs.clone()
        idxs_masked[~alive] = n  # sentinel

        sorted_idxs = torch.sort(idxs_masked, dim=1)[0]  # [B, n] ascending with sentinel at end
        topk_keep = sorted_idxs[:, :k]                  # [B, k]
        topk_keep[topk_keep == n] = -1
        return topk_keep
    
    def fps_from_cosine_sim(self,cos_sim, k):
        """
        Batched FPS on cosine similarity matrices.
        Assumes cos_sim computed from L2-normalized embeddings (no abs).
        Returns indices [B, k] in selection order.
        """
        device = cos_sim.device
        B, n, _ = cos_sim.shape
        if k <= 0:
            return torch.empty((B, 0), dtype=torch.long, device=device)
        if k >= n:
            return torch.arange(n, device=device).unsqueeze(0).expand(B, -1).clone()

        # convert similarity -> distance in [0,2] for cos in [-1,1]
        distance = 1.0 - cos_sim.clone()   # higher -> farther

        # seed with farthest pair (max distance among i<j)
        idx = torch.arange(n, device=device)
        # avoid self-pairs by making diagonal tiny negative
        distance[:, idx, idx] = -1.0

        tri = torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)
        masked = distance.clone()
        masked[:, ~tri] = float('-inf')        # only keep upper-tri values
        flat = masked.view(B, -1)
        max_pair = flat.argmax(dim=1)          # pair with largest distance
        r = max_pair // n
        c = max_pair % n

        # bookkeeping
        selected = torch.zeros((B, n), dtype=torch.bool, device=device)
        batch_idx = torch.arange(B, device=device)
        selected[batch_idx, r] = True
        selected[batch_idx, c] = True

        result = torch.full((B, k), -1, dtype=torch.long, device=device)
        result[:, 0] = r
        if k > 1:
            result[:, 1] = c

        selected_count = 2 if k > 1 else 1

        # iterative FPS: for each unselected i compute min_j d(i,j) over selected j,
        # then pick argmax of that min-distance
        POS_INF = float('inf')
        NEG_INF = float('-inf')

        for t in range(selected_count, k):
            # pair_mask[b,i,j] == True iff i is UNselected and j is SELECTED
            sel_mask = selected.unsqueeze(1)        # [B,1,n]
            unsel_mask = (~selected).unsqueeze(2)   # [B,n,1]
            pair_mask = unsel_mask & sel_mask       # [B,n,n]

            # For min we want non-(unselected->selected) entries to be +inf so they don't reduce the min
            # (only selected j's for each candidate i remain finite)
            distance_masked = torch.where(pair_mask, distance, torch.full_like(distance, POS_INF))  # [B,n,n]

            # minimum distance to selected set for each candidate i
            min_dist_to_selected, _ = distance_masked.min(dim=2)   # [B,n]

            # prevent selecting already selected entries: set their min-dist to NEG_INF so they won't be argmax
            min_dist_to_selected = min_dist_to_selected.masked_fill(selected, NEG_INF)

            # now pick candidate with maximum of those minima
            next_idx = min_dist_to_selected.argmax(dim=1)   # [B]
            result[:, t] = next_idx
            selected[batch_idx, next_idx] = True

        return result


    # def get_random_topk_indices(self, cos_sim, k):
    #     """ Selects k random token indices for each sample in the batch.

    #     Args:
    #         cos_sim: A tensor of shape (batch, num_tokens, num_tokens)
    #         k: The number of tokens to select randomly.
    #     Returns:
    #         A tensor of shape (batch, k) containing k random indices for each sample.
    #     """
    #     batch_size, num_tokens, _ = cos_sim.shape
    #     rand_indices = torch.stack([torch.randperm(num_tokens)[:k] for _ in range(batch_size)])
    #     return rand_indices


    def get_random_topk_indices(self, cos_sim, k, keys):
        """
        Deterministic per-image random indices (same every epoch/run for the same key).
        Args:
            cos_sim: Tensor of shape (batch, num_tokens, num_tokens)
            k: number of tokens to select
            keys: iterable of unique strings/IDs for each sample in the batch (len == batch)
        Returns:
            Tensor of shape (batch, k) on the same device as cos_sim
        """
        batch_size, num_tokens, _ = cos_sim.shape
        device = cos_sim.device

        if keys is None or len(keys) != batch_size:
            raise ValueError("len(keys) must equal batch_size")

        selected_indices = []
        for key in keys:
            # stable 32-bit seed from key
            seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)

            # local generator -> does not change global RNG state
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)

            # deterministic permutation for this key
            idx = torch.randperm(num_tokens, generator=gen, device=device)[:k]
            selected_indices.append(idx)

        return torch.stack(selected_indices, dim=0)  # (batch, k)

    def _log_mask_visualization(self, raw_input, keep_mask, patch_size, num_patches_h, num_patches_w):
        """
        Log a side-by-side visualization of raw vs masked image for the first sample.
        This is for debugging/inspection and does not affect the forward pass.
        """
        # Only log once to avoid flooding W&B
        if not getattr(self, "plot_img", False):
            return
        if getattr(self, "plot_cnt", 0) > 0:
            return
        if not wandb.run:
            return

        try:
            # Take first sample
            raw_img = raw_input[0].detach().cpu().numpy()  # (C, H, W)
            if raw_img.shape[0] > 3:
                raw_img = raw_img[:3, :, :]
            # Normalize to [0, 1] for display
            raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
            raw_img = raw_img.transpose(1, 2, 0)  # (H, W, C)

            patch_mask = keep_mask[0].detach().cpu().numpy()  # (P,)
            total_patches = patch_mask.shape[0]

            masked_img = np.zeros_like(raw_img)

            for idx in range(total_patches):
                h_idx = idx // num_patches_w
                w_idx = idx % num_patches_w

                h_start = h_idx * patch_size
                h_end = (h_idx + 1) * patch_size
                w_start = w_idx * patch_size
                w_end = (w_idx + 1) * patch_size

                if patch_mask[idx]:
                    masked_img[h_start:h_end, w_start:w_end] = raw_img[h_start:h_end, w_start:w_end]
                else:
                    masked_img[h_start:h_end, w_start:w_end] = 0.0  # black out masked patches

            # Concatenate raw and masked horizontally into a single panel
            panel = np.concatenate([raw_img, masked_img], axis=1)  # (H, 2W, C)
            wandb.log({"Raw vs Masked (RViT)": wandb.Image(panel)})

            self.plot_cnt = getattr(self, "plot_cnt", 0) + 1
        except Exception as e:
            # Do not fail training because of visualization issues
            if getattr(self, "plot_img", False):
                print(f"Mask visualization failed: {e}")

    
        # def visualization_input(self, raw_input, raw_label, mask, hier_mask, patch_size, num_patches_w, total_patches):
        # #Convert raw input to CPU numpy array
        #     raw_img = raw_input[0].detach().cpu().numpy()  # take first sample
        #     if raw_img.shape[0] > 3:
        #         raw_img = raw_img[:3, :, :]
        #     raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
        #     raw_img = raw_img.transpose(1, 2, 0)

        #     # Create masked image with patches in original positions
        #     masked_img = np.zeros_like(raw_img)
        #     masked_label = np.zeros_like(raw_label)
        #     patch_label_mask = hier_mask[0].cpu().numpy()  # Use first sample's mask
        #     patch_mask = mask[0].cpu().numpy()  # Use first sample's mask

        #     for idx in range(total_patches):
        #         h_idx = idx // num_patches_w
        #         w_idx = idx % num_patches_w
                
        #         h_start = h_idx * patch_size
        #         h_end = (h_idx + 1) * patch_size
        #         w_start = w_idx * patch_size
        #         w_end = (w_idx + 1) * patch_size
                
        #         if patch_mask[idx]:
        #             # Keep original patch
        #             masked_img[h_start:h_end, w_start:w_end] = raw_img[h_start:h_end, w_start:w_end]
        #             masked_label[h_start:h_end, w_start:w_end] = raw_label[h_start:h_end, w_start:w_end]
        #         else:
        #             # Black out masked patches
        #             masked_img[h_start:h_end, w_start:w_end] = 0.0  # Use black for masked patches
        #             masked_label[h_start:h_end, w_start:w_end] = -1.0  # Use black for masked patches
                            
        #     # Plot side by side
        #     #fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        #     fig, axes = plt.subplots(1, 4, figsize=(16, 6))
        #     axes[0].imshow(raw_img)
        #     axes[0].set_title("Raw Input")
        #     axes[0].axis("off")
            
        #     axes[1].imshow(masked_img)
        #     axes[1].set_title(f"Masked Image ({self.implementation})")
        #     axes[1].axis("off")
            
        #     if self.hierarchical_features:
        #         axes[2].imshow(raw_label)
        #         axes[2].set_title(f"Raw Label ({self.implementation})")
        #         axes[2].axis("off")

        #         axes[3].imshow(masked_label)
        #         axes[3].set_title(f"Masked Label ({self.implementation})")
        #         axes[3].axis("off")

        #         plt.tight_layout()
        #     #plt.savefig(f"masked_vs_raw_{self.implementation}.png", dpi=300)
        #     plt.close()

        #     if wandb.run:
        #         wandb.log({
        #             "Raw vs Masked vs Raw Label vs Masked Label": wandb.Image(fig),
        #             #"Retained Patches": patch_mask.sum() / total_patches
        #         })
    
    def forward_mask_in_image_space(self, x, keys=None, label=None,verbose=False):
        """
        Mask patches based on cosine similarity and retain the most important ones.
        Masked patches are replaced with padding.
        """
        raw_input = x.clone()

        hierarchical_mask = None
        
        if self.training and self.mask_threshold <= 1.0 and self.retain_percent <= 100:
            patch_size = self.model.patch_embed.patch_size[0]

            # Ensure input dimensions are multiples of patch size
            H, W = x.shape[2], x.shape[3]
            pad_h, pad_w = (patch_size - H % patch_size) % patch_size, (patch_size - W % patch_size) % patch_size
            x = F.pad(x, (0, pad_w, 0, pad_h))

            # Extract patches
            patches = einops.rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)
            
            #SSIM similarity
            # if patches is (B, P, patch_flat), convert to (B, P, C, ph, pw)
            B, P, flat = patches.shape
            C = x.shape[1]    # original image channels before rearrange (make sure you have it)
            ph, pw = patch_size, patch_size
            patches_img = patches.view(B, P, C, ph, pw)   # adjust if dimension order differs
            #ssim = self.batch_pairwise_ssim(patches_img, ph, pw)   # returns (B,P,P) in [0,1]
            # Compute cosine similarity
            #cos_sim = self.batch_cos_similarity(patches)
            cos_sim = torch.abs(self.batch_cos_similarity(patches))
            # Get binary mask (True = keep, False = mask)
            keep_mask = self.get_mask_vectorised(cos_sim)
            #keep_mask = self.get_mask_vectorised(ssim)
             # Store original image dimensions for visualization
            orig_H, orig_W = H + pad_h, W + pad_w
            num_patches_h, num_patches_w = orig_H // patch_size, orig_W // patch_size

            

        # Extract actual token embeddings
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)

        # Handle CLS token separately
        cls_token = None
        if self.model.global_pool == 'token':
            cls_token, x = x[:, :1, :], x[:, 1:]  # Separate CLS token

        total_patches = x.shape[1]  # Exclude CLS token count


        final_mask = torch.zeros(x.shape[0], total_patches, dtype=torch.bool, device=x.device)

       # if self.training:
            #final_mask = mask.clone()

        #3 Options: 1st-topx%, 2nd-median, 3rd-max_len
        if self.implementation == "topx%":
         
            if self.training and self.retain_percent <= 100:
                #keep_indices = self.get_topk_indices_mean_similarity(cos_sim, int(cos_sim.shape[1] * (self.retain_percent / 100)))
                #keep_indices = self.get_topk_indices_redundancy_based(cos_sim, int(cos_sim.shape[1] * (self.retain_percent / 100)))
                keep_indices = self.fps_from_cosine_sim(cos_sim, int(cos_sim.shape[1] * (self.retain_percent / 100)))
                # Ensure we select the same number of patches across all samples
                k = keep_indices.shape[1]  # Number of patches to retain
                
                # Sort indices to ensure consistent ordering across different selection approaches
                keep_indices, _ = torch.sort(keep_indices, dim=1)
                
                # Gather selected patches (reorder for < 100% retention)
                batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(-1).expand(-1, k)
                orig_shape = x.shape
                x = x[batch_indices, keep_indices]  # Efficient batch gather
                
                hierarchical_mask = torch.zeros(orig_shape[:-1], dtype=torch.bool,device=x.device)
                #batch_indices = torch.arange(keep_indices.size(0), device=keep_mask.device).unsqueeze(-1)
                
                batch_indices = torch.arange(keep_indices.size(0), device=keep_indices.device).unsqueeze(-1)
                hierarchical_mask[batch_indices, keep_indices] = True
                final_mask[batch_indices, keep_indices] = True

                if verbose:
                    print(f"Cos sim shape: {cos_sim.shape}, Selected indices: {keep_indices.shape}, Output shape: {x.shape}")

                if wandb.run:
                    wandb.log({"unmasked_patches_ratio": k / total_patches})
        
        elif self.implementation == "random_topx%":

            if self.training and self.retain_percent <= 100:
                # Get random top-k indices
                if keys is None:
                    raise ValueError("keys must be provided for 'random_topx%' implementation")
                keep_indices = self.get_random_topk_indices(cos_sim, int(cos_sim.shape[1] * (self.retain_percent / 100)), keys)
                # Ensure we select the same number of patches across all samples
                k = keep_indices.shape[1]
                # Gather selected patches
                batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(-1).expand(-1, k)
                orig_shape = x.shape
                #print(orig_shape)
                x = x[batch_indices, keep_indices]  # Efficient batch gather
                hierarchical_mask = torch.zeros(orig_shape[:-1], dtype=torch.bool, device=x.device)
                #print(hierarchical_mask.shape)
                batch_indices = torch.arange(keep_indices.size(0), device=x.device).unsqueeze(-1)
                
                # Set the top indices to True
                hierarchical_mask[batch_indices, keep_indices] = True
                final_mask[batch_indices, keep_indices] = True  

                if verbose:
                    print(f"Cos sim shape: {cos_sim.shape}, Selected indices: {keep_indices.shape}, Output shape: {x.shape}")

                if wandb.run:
                    wandb.log({"unmasked_patches_ratio": k / orig_shape[1]})

        elif self.implementation == "max_len":
            if self.training and self.mask_threshold <= 1.0:

                keep_mask = keep_mask[:, :total_patches]  # Ensure matching shape

                # **Vectorized Retained Tokens Selection**
                lens = keep_mask.sum(dim=1)  # Get number of unmasked patches per image
               
                max_len = min(lens.max().item(), total_patches)
                mean_len = lens.float().mean().item()
                #padded tensor efficient creation
                padded_x = torch.full((x.shape[0], max_len, x.shape[2]), self.pad_value, device=x.device, dtype=x.dtype)

                #Use Masking to Copy Retained Tokens
                mask = torch.arange(total_patches, device=x.device).unsqueeze(0) < lens.unsqueeze(1)
                padded_x[mask[:, :max_len]] = x[keep_mask]  # Efficiently copy retained tokens

                x = padded_x  # Update x with masked patches
              
                hierarchical_mask = keep_mask
                final_mask = keep_mask
                
                #if self.attention_mask:
                token_attention_mask = torch.zeros((x.shape[0], max_len), dtype=torch.bool, device=x.device)
                token_attention_mask[~mask[:, :max_len]] = True  # True for masked-out tokens

                if verbose:
                    print(f"Cos sim shape: {cos_sim.shape}, Mask shape: {keep_mask.shape}, Unmasked shape: {x.shape}, Max len: {max_len}")

                if wandb.run:
                    wandb.log({"unmasked_patches_ratio":  mean_len / total_patches})


        elif self.implementation == "median":
            if self.training and self.mask_threshold <= 1.0:

                keep_mask = keep_mask[:, :total_patches]  # Ensure matching shape     
                #Mask based on similarity
                retained_tokens = [x[i, keep_mask[i]] for i in range(x.shape[0])]
                
                #Get median length
                lengths = torch.tensor([tokens.shape[0] for tokens in retained_tokens], device=x.device)
                median_len = int(torch.median(lengths).item())
                
                #Calc topk indices
                topk_indices = self.get_topk_indices_redundancy_based(cos_sim,median_len)
                #topk_indices = self.get_topk_indices_mean_similarity(cos_sim,median_len)
                # Sort indices to ensure consistent ordering
                topk_indices, _ = torch.sort(topk_indices, dim=1)
                batch_indices = torch.arange(x.shape[0], device=x.device)[:, None]
                
                x = x[batch_indices,topk_indices]
                hierarchical_mask = torch.zeros_like(keep_mask, dtype=torch.bool,device=x.device)
                batch_indices = torch.arange(topk_indices.size(0), device=keep_mask.device).unsqueeze(-1)

                # Set the top indices to True
                hierarchical_mask[batch_indices, topk_indices] = True
                final_mask[batch_indices, topk_indices] = True   
                
                if verbose:
                    print(f"Cos sim shape: {cos_sim.shape}, Mask shape: {keep_mask.shape}, Unmasked shape: {x.shape}, Median length: {median_len}")

                if wandb.run:
                    wandb.log({"unmasked_patches_ratio": median_len / total_patches})

            
        else: #vit
                if wandb.run:
                    wandb.log({"total_patches": total_patches})

        # Re-add CLS token if necessary
        if self.model.global_pool == 'token' and cls_token is not None:
                cls_token = cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls_token, x], dim=1)
    
  

        # Remove CLS token
        if self.training and self.attention_mask and not self.hierarchical_features:
                cls_mask = torch.zeros((x.shape[0], 1), device=x.device, dtype=torch.bool)
                token_attention_mask = torch.cat((cls_mask, token_attention_mask), dim=1)

        # after all branches where final_mask is filled
        if self.training and self.mask_threshold <= 1.0 and self.retain_percent <= 100 and wandb.run:
                self._log_mask_visualization(raw_input, final_mask, patch_size, num_patches_h, num_patches_w)

        # Forward through the model
        x = self.model.norm_pre(x)

        """
        if self.training and self.attention_mask:
            token_attention_mask = token_attention_mask[:, None, None, :]  # Shape it for attention layers
            for module in self.model.blocks:
                x = module(x, token_attention_mask)
        else:
            x = self.model.blocks(x)
        """

        
        if self.training and self.attention_mask:
            token_attention_mask = token_attention_mask[:, None, None, :]  # Shape it for attention layers
            if not self.hierarchical_features:
                for module in self.model.blocks:
                    x = module(x, token_attention_mask)
            else:
                x = x[:, 1:]  # Remove CLS token
                
                for block in self.stage1_blocks:
                    features = block(x, token_attention_mask)
                stage1 = features

                # Stage 2
                for block in self.stage2_blocks:
                    features = block(features, token_attention_mask)
                stage2 = features

                # Stage 3
                for block in self.stage3_blocks:
                    features = block(features, token_attention_mask)
                stage3 = features

                # Stage 4
                for block in self.stage4_blocks:
                    features = block(features, token_attention_mask)
                stage4 = features
                x = features
        else:
            if not self.hierarchical_features:
                x = self.model.blocks(x)
            else:
                x = x[:,1:]  # Remove CLS token
                for block in self.stage1_blocks:
                    features = block(x)
                stage1 = features

                # Stage 2
                for block in self.stage2_blocks:
                    features = block(features)
                stage2 = features

                # Stage 3
                for block in self.stage3_blocks:
                    features = block(features)
                stage3 = features

                # Stage 4
                for block in self.stage4_blocks:
                    features = block(features)
                stage4 = features
                x = features

        x = self.model.norm(x)

        if not self.hierarchical_features:
            return x, None, None
        else:
            return [stage1, stage2, stage3, stage4], hierarchical_mask, final_mask

    
    #SEGMENTATION
    def forward(self, x, keys=None, label=None):
        if self.mask_in_token_space:
            x = self.forward_mask_in_token_space(x,verbose=False)
        else:
                x, hierarchical_mask, final_mask = self.forward_mask_in_image_space(x, keys, label, verbose=False)
        if not self.hierarchical_features:
            return self.model.forward_head(x)
        else:
            return x, hierarchical_mask, final_mask


if __name__ == '__main__':
    model = RViT(num_classes=14, backbone='vit_base_patch16_224', image_size=128, num_channels=14, mask_threshold=0.9, mask_in_token_space=False, implementation="max_len", pad_value = 0.0, retain_percent=60, attention_mask=True)
    
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