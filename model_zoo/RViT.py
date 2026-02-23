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


class RViT(nn.Module):
    def __init__(self, num_classes, backbone, image_size, num_channels, pretrained=True,
                 mask_threshold=0.9, mask_in_token_space=True, implementation="max_len",
                 pad_value=0.0, retain_percent=0, attention_mask=False,
                 hierarchical_features=False, apply_mask_in_infer=False):
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
            apply_mask_in_infer: If True, apply the masking logic during the
                                 testing phase (inference), but NOT during validation.
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
        # Controls whether masking is used during inference (test phase only)
        self.apply_mask_in_infer = apply_mask_in_infer
        self.plot_img = True
        self.plot_cnt = 0

        if not self.attention_mask:
            self.model = timm.create_model(backbone,in_chans = num_channels, num_classes=num_classes, img_size=image_size, dynamic_img_pad=True, dynamic_img_size=True,pretrained=pretrained)
            #self.model.patch_embed.set_input_size(image_size, patch_size=patch_size)
        else:
            from model_zoo import MaskAttentionViT as MaskAttentionViT

            if self.implementation in ["topx%", "vit", "random_topx%", "different_random_topx%"]:
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
              
        if self.implementation not in ["max_len", "topx%", "vit", "random_topx%", "different_random_topx%"]:
            raise ValueError("Invalid or missing `implementation`. "
                                "Choose from: [max_len, topx%, vit, random_topx%, different_random_topx%]")
        
        if (self.implementation == "topx%" or self.implementation == "random_topx%" or self.implementation == "different_random_topx%") and (self.retain_percent <= 0 or self.retain_percent > 100):
            raise ValueError("Value of retain_percent must be greater than 0 and less than or equal to 100 for topx% implementation.")
        


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

        # Count the number of tokens retained for each sample
        retained_counts = keep_mask.sum(dim=1).tolist()
        
        # Get max length
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
        # Decide whether to apply masking:
        # - always in training
        # - optionally in testing (inference) if apply_mask_in_infer is True
        is_testing = getattr(self, "testing", False)
        apply_infer = getattr(self, "apply_mask_in_infer", False)
        use_mask = self.training or (is_testing and apply_infer)

        if use_mask:
            x = self.calc_mask_vectorised(x,False)

        

        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        return x

    
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


    def get_different_random_topk_indices(self, cos_sim, k):
        """Selects k random token indices per sample.

        Args:
            cos_sim: A tensor of shape (batch, num_tokens, num_tokens)
            k: The number of tokens to select randomly.
            keys: iterable of unique strings/IDs for each sample (len == batch); used to seed RNG per sample.
        Returns:
            A tensor of shape (batch, k) containing k random indices for each sample.
        """
        batch_size, num_tokens, _ = cos_sim.shape
        rand_indices = torch.stack([torch.randperm(num_tokens)[:k] for _ in range(batch_size)])
        return rand_indices


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
            # Handle both string and Tensor keys
            if isinstance(key, torch.Tensor):
                # Convert tensor to string representation for hashing
                # Use a stable representation (e.g., convert to bytes or use item() if scalar)
                if key.numel() == 1:
                    key_str = str(key.item())
                else:
                    # For multi-element tensors, use a hash of the tensor values
                    # Convert to numpy first, then to bytes for stable hashing
                    key_str = str(key.detach().cpu().numpy().tobytes())
            elif isinstance(key, (bytes, bytearray)):
                key_str = key.decode('utf-8', errors='ignore')
            else:
                key_str = str(key)
            
            # stable 32-bit seed from key
            seed = int(hashlib.md5(key_str.encode()).hexdigest(), 16) % (2**32)

            # local generator -> does not change global RNG state
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)

            # deterministic permutation for this key
            idx = torch.randperm(num_tokens, generator=gen, device=device)[:k]
            selected_indices.append(idx)

        return torch.stack(selected_indices, dim=0)  # (batch, k)

    def log_mask_visualization(self, raw_input, keep_mask, patch_size, num_patches_h, num_patches_w):
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

    
    def forward_mask_in_image_space(self, x, keys=None, label=None,verbose=False):
        """
        Mask patches based on cosine similarity and retain the most important ones.
        Masked patches are replaced with padding.
        """
        raw_input = x.clone()

        hierarchical_mask = None

        is_testing = getattr(self, "testing", False)
        apply_infer = getattr(self, "apply_mask_in_infer", False)
        use_mask = self.training or (is_testing and apply_infer)
        
        # Debug print for masking logic
        if is_testing and not hasattr(self, "_debug_printed_masking"):
            print(f"[DEBUG] forward_mask_in_image_space: use_mask={use_mask}, "
                  f"mask_threshold={self.mask_threshold}, retain_percent={self.retain_percent}, "
                  f"training={self.training}, is_testing={is_testing}, apply_infer={apply_infer}")
            self._debug_printed_masking = True
        
        if use_mask and self.mask_threshold <= 1.0 and self.retain_percent <= 100:
            patch_size = self.model.patch_embed.patch_size[0]

            # Ensure input dimensions are multiples of patch size
            H, W = x.shape[2], x.shape[3]
            pad_h, pad_w = (patch_size - H % patch_size) % patch_size, (patch_size - W % patch_size) % patch_size
            x = F.pad(x, (0, pad_w, 0, pad_h))

            # Extract patches
            patches = einops.rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)
            
            
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
        x_before_drop = x.clone()  # Save for debugging
        x = self.model.patch_drop(x)
        
        # Debug: check if patch_drop changed the number of tokens
        if is_testing and not hasattr(self, "_debug_printed_patch_drop"):
            if hasattr(self.model, 'patch_drop') and hasattr(self.model.patch_drop, 'drop_prob'):
                drop_prob = self.model.patch_drop.drop_prob if hasattr(self.model.patch_drop, 'drop_prob') else 0.0
                print(f"[DEBUG] patch_drop: drop_prob={drop_prob}, "
                      f"tokens before drop={x_before_drop.shape[1]}, after drop={x.shape[1]}")
            else:
                print(f"[DEBUG] patch_drop: tokens={x.shape[1]} (no drop_prob attribute)")
            self._debug_printed_patch_drop = True

        # Handle CLS token separately
        cls_token = None
        if self.model.global_pool == 'token':
            cls_token, x = x[:, :1, :], x[:, 1:]  # Separate CLS token

        total_patches = x.shape[1]  # Exclude CLS token count


        final_mask = torch.zeros(x.shape[0], total_patches, dtype=torch.bool, device=x.device)
        token_attention_mask = None  # Set only in max_len; used so padding is not attended (train and eval)

       # if self.training:
            #final_mask = mask.clone()

        #3 Options: 1st-topx%, 2nd-different_random_topx%, 3rd-random_topx%, 4th-max_len
        if self.implementation == "topx%":
         
            if use_mask and self.retain_percent <= 100:
                # Debug: verify cos_sim exists and show patch counts
                if is_testing and not hasattr(self, "_debug_printed_topx"):
                    print(f"[DEBUG] topx% masking: cos_sim exists={('cos_sim' in locals())}, "
                          f"cos_sim.shape={cos_sim.shape if 'cos_sim' in locals() else 'N/A'}, "
                          f"total_patches={total_patches}, "
                          f"x.shape (tokens)={x.shape}, "
                          f"target_k={int(cos_sim.shape[1] * (self.retain_percent / 100)) if 'cos_sim' in locals() else 'N/A'}")
                    self._debug_printed_topx = True
                
                # Verify cos_sim was computed (should exist from earlier block)
                if 'cos_sim' not in locals():
                    print(f"[ERROR] cos_sim not found! Masking cannot proceed.")
                    # Fallback: skip masking
                else:
                    #keep_indices = self.get_topk_indices_mean_similarity(cos_sim, int(cos_sim.shape[1] * (self.retain_percent / 100)))
                    #keep_indices = self.get_topk_indices_redundancy_based(cos_sim, int(cos_sim.shape[1] * (self.retain_percent / 100)))
                    target_k = int(cos_sim.shape[1] * (self.retain_percent / 100))
                    keep_indices = self.fps_from_cosine_sim(cos_sim, target_k)
                    # Ensure we select the same number of patches across all samples
                    k = keep_indices.shape[1]  # Number of patches to retain
                    
                    # Sort indices to ensure consistent ordering across different selection approaches
                    keep_indices, _ = torch.sort(keep_indices, dim=1)
                    
                    if is_testing and not hasattr(self, "_debug_printed_topx_result"):
                        print(f"[DEBUG] topx% masking RESULT: Selected {k} patches out of {cos_sim.shape[1]} "
                              f"({100*k/cos_sim.shape[1]:.1f}%), "
                              f"x.shape before selection={x.shape}, "
                              f"keep_indices.shape={keep_indices.shape}, "
                              f"keep_indices sample[0,:10]={keep_indices[0,:10].cpu().tolist() if keep_indices.numel() > 0 else 'empty'}")
                        self._debug_printed_topx_result = True
                    
                    # Gather selected patches (reorder for < 100% retention)
                    batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(-1).expand(-1, k)
                    orig_shape = x.shape
                    x = x[batch_indices, keep_indices]  # Efficient batch gather
                    
                    if is_testing and not hasattr(self, "_debug_printed_topx_final"):
                        print(f"[DEBUG] topx% masking FINAL: x.shape after selection={x.shape}, "
                              f"retained {k}/{total_patches} tokens ({100*k/total_patches:.1f}%)")
                        self._debug_printed_topx_final = True
                    
                    hierarchical_mask = torch.zeros(orig_shape[:-1], dtype=torch.bool,device=x.device)
                    batch_indices = torch.arange(keep_indices.size(0), device=keep_indices.device).unsqueeze(-1)
                    hierarchical_mask[batch_indices, keep_indices] = True
                    final_mask[batch_indices, keep_indices] = True

                    if verbose:
                        print(f"Cos sim shape: {cos_sim.shape}, Selected indices: {keep_indices.shape}, Output shape: {x.shape}")

                    if wandb.run:
                        wandb.log({"unmasked_patches_ratio": k / total_patches})

        elif self.implementation == "different_random_topx%":
            if use_mask and self.retain_percent <= 100:
                # Get random top-k indices
                # If keys are None (e.g., in test phase), generate deterministic keys from batch indices
                if keys is None:
                    if is_testing:
                        # Generate deterministic keys for test phase: "test_sample_0", "test_sample_1", etc.
                        batch_size = cos_sim.shape[0]
                        keys = [f"test_sample_{i}" for i in range(batch_size)]
                    else:
                        raise ValueError("keys must be provided for 'different_random_topx%' implementation in training")
                
                keep_indices = self.get_different_random_topk_indices(cos_sim, int(cos_sim.shape[1] * (self.retain_percent / 100)))
                # Sort so sequence is in spatial order; otherwise decoder gets scrambled layout and IoU drops
                keep_indices, _ = torch.sort(keep_indices, dim=1)
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
        
        elif self.implementation == "random_topx%":

            if use_mask and self.retain_percent <= 100:
                # Get random top-k indices
                # If keys are None (e.g., in test phase), generate deterministic keys from batch indices
                if keys is None:
                    if is_testing:
                        # Generate deterministic keys for test phase: "test_sample_0", "test_sample_1", etc.
                        batch_size = cos_sim.shape[0]
                        keys = [f"test_sample_{i}" for i in range(batch_size)]
                    else:
                        raise ValueError("keys must be provided for 'random_topx%' implementation in training")
                
                keep_indices = self.get_random_topk_indices(cos_sim, int(cos_sim.shape[1] * (self.retain_percent / 100)), keys)
                # Sort so sequence is in spatial order; otherwise decoder gets scrambled layout and IoU drops
                keep_indices, _ = torch.sort(keep_indices, dim=1)
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
            if use_mask and self.mask_threshold <= 1.0:

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

            
        else: #vit
                if wandb.run:
                    wandb.log({"total_patches": total_patches})

        # Re-add CLS token if necessary
        if self.model.global_pool == 'token' and cls_token is not None:
                cls_token = cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls_token, x], dim=1)
    
  

        # Remove CLS token
        if use_mask and self.attention_mask and not self.hierarchical_features:
                cls_mask = torch.zeros((x.shape[0], 1), device=x.device, dtype=torch.bool)
                token_attention_mask = torch.cat((cls_mask, token_attention_mask), dim=1)

        # after all branches where final_mask is filled
        if use_mask and self.mask_threshold <= 1.0 and self.retain_percent <= 100 and wandb.run:
                self.log_mask_visualization(raw_input, final_mask, patch_size, num_patches_h, num_patches_w)

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

        
        if token_attention_mask is not None:
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
            return x, None, final_mask  # return final_mask so visualization can show kept/masked patches
        else:
            return [stage1, stage2, stage3, stage4], hierarchical_mask, final_mask

    
    #SEGMENTATION
    def forward(self, x, keys=None, label=None):
        # FinetunerClassification passes (image, spectral_keys); use image tensor only for masking/backbone.
        if isinstance(x, (tuple, list)):
            x = x[0]
        # Lightweight debug print to confirm RViT is used in test phase
        if getattr(self, "testing", False) and not hasattr(self, "_debug_printed_forward"):
            print(f"[DEBUG] RViT forward in testing mode. "
                  f"implementation={self.implementation}, "
                  f"retain_percent={self.retain_percent}, "
                  f"mask_in_token_space={self.mask_in_token_space}, "
                  f"apply_mask_in_infer={getattr(self, 'apply_mask_in_infer', None)}")
            self._debug_printed_forward = True

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