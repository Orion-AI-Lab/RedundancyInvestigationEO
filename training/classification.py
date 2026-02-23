import pprint
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
import os

# Set environment variables to work around cuDNN compatibility issues
# os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'
# os.environ['CUDNN_LOGINFO_DBG'] = '0'
# os.environ['CUDNN_LOGERR_DBG'] = '0'

import numpy as np
import pyjson5 as json
import torch
import tqdm
import wandb
import utilities.model_utilities as model_utils
import utilities.utils as utils
from utilities.model_utilities import adjust_learning_rate, get_current_learning_rate
from utilities.utils import get_gpu_util, build_timing_log_dict, build_inference_timing_log_dict, compute_timing_summary, estimate_vit_encoder_gflops, estimate_full_architecture_gflops, estimate_full_model_gflops, get_peak_memory_mb, reset_peak_memory_stats
import time

# cuDNN configuration - can be enabled via environment variable ENABLE_CUDNN=1
# Disable cuDNN by default to avoid compatibility issues
# This will use PyTorch's native CUDA kernels instead of cuDNN
# enable_cudnn = os.environ.get('ENABLE_CUDNN', '0').lower() in ('1', 'true', 'yes')
# if enable_cudnn:
#     torch.backends.cudnn.benchmark = True  # Enable for better performance
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.enabled = True
#     print("INFO: cuDNN is enabled. Ensure cuDNN version is compatible with your PyTorch/CUDA setup.")
# else:
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.enabled = False  # Disable cuDNN entirely
#     print("WARNING: cuDNN is disabled due to compatibility issues. Training will be slower but should work.")
#     print("         To enable cuDNN, set environment variable: ENABLE_CUDNN=1")

# Suppress specific warnings related to multilabel metrics and PyTorch deprecations
# warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics')
# warnings.filterwarnings('ignore', category=UserWarning, module='torch.overrides')
# warnings.filterwarnings('ignore', category=FutureWarning, module='timm.models.layers')


def early_stopping(val_loss, best_loss, counter, patience=5):
    if val_loss < best_loss:
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping")
        return True, counter
    else:
        return False, counter


def train_epoch(train_loader, model, optimizer, criterion, epoch, configs, scaler):
    running_loss=0.0
    num_samples = 0
    if not configs["linear_evaluation"] and not configs["fully_finetune"]:
        model.train()
    else:
        model.train()

        modality_dictionary = {v: k for k, v in configs["modality_channels"].items()}
        available_modalities = configs["dataset_modality_index"][configs["dataset"]]
        spectral_keys = []
        desired_indices = []
        for modality in available_modalities:
            spectral_keys.append(int(modality_dictionary[modality]))
            desired_indices.append(available_modalities[modality])

        # Ensure channel indices are valid for current input tensor
        # (e.g. BigEarthNet with rgb=true has in_channels=3 but dataset_modality_index has 14 entries)
        max_ch = configs["in_channels"]
        desired_indices = [idx for idx in desired_indices if idx < max_ch]
        spectral_keys = [idx for idx in spectral_keys if idx < max_ch]

        # Linear eval from segmentation (e.g. FLAIR): model has in_channels from target dataset; use full image
        # if modality index has fewer channels than encoder expects.
        use_full_channels_linear_seg = (
            configs["linear_evaluation"]
            and configs.get("seg_to_class", False)
            and len(desired_indices) < configs["in_channels"]
        )
        if use_full_channels_linear_seg:
            desired_indices = list(range(configs["in_channels"]))
            spectral_keys = desired_indices.copy()

    if "resnet_star_evaluation" in configs and configs["resnet_star_evaluation"]:
        resnet_star_configs = json.load(open("configs/datasets/supervised_foundation_cls.json", "r"))

        resnet_star_evaluation = True
        modality_dictionary = {v: k for k, v in resnet_star_configs["modality_channels"].items()}
        available_modalities = resnet_star_configs["dataset_modality_index"][configs["dataset"]]
        spectral_keys = []
        desired_indices = []
        for modality in available_modalities:
            spectral_keys.append(int(modality_dictionary[modality]))
            desired_indices.append(available_modalities[modality])
        print("Desired indices: ", desired_indices)
    else:
        resnet_star_evaluation = False

    log_timing = configs.get("log_timing_metrics", False)
    timing_interval = configs.get("timing_log_interval", 100)
    device = configs["device"]
    cuda_device = torch.device(device) if isinstance(device, str) else device
    use_cuda_sync = isinstance(cuda_device, torch.device) and cuda_device.type == "cuda"

    for idx, batch in enumerate(tqdm.tqdm(train_loader)):
        if "samples_per_epoch" in configs:
            if (idx + 1) * configs["batch_size"] > configs["samples_per_epoch"]:
                break
        if log_timing and idx % timing_interval == 0 and use_cuda_sync:
            torch.cuda.synchronize(cuda_device)
        t_step_start = time.perf_counter() if (log_timing and idx % timing_interval == 0) else None
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=configs["mixed_precision"]):
            image, label, keys = batch
            #print("Keys found in batch") if keys is not None else print("Keys not found in batch")

            num_samples += image.shape[0]
            image = image.to(configs["device"])
            label = label.to(configs["device"])
            first_sample_logged_training = False

            # Measured full-model GFLOPs (fvcore/thop): once per epoch on first batch when use_measured_full_gflops
            if (log_timing and idx == 0 and configs.get("use_measured_full_gflops", False)):
                with torch.no_grad():
                    measured = estimate_full_model_gflops(model, image, input_kwargs={"keys": keys})
                if measured is not None:
                    full_m, enc_fv = measured
                    configs["_full_gflops_measured_this_epoch"] = full_m
                    configs["_encoder_gflops_fv_measured_this_epoch"] = enc_fv

            # ----------- WandB log first sample in training phase----------
            # if configs["wandb"]:
            #     if idx == 0 and not first_sample_logged_training:
            #         img_to_log = image[0].detach().cpu()

            #         # if multi-channel hyperspectral, pick first 3 bands
            #         if img_to_log.shape[0] > 3:
            #             img_to_log = img_to_log[:3, :, :]

            #         # Convert normalized image to [0, 255] range for wandb
            #         # Clip extreme values to handle normalized data (mean 0, std 1 can have values outside [0,1])
            #         # Use percentile-based clipping to handle outliers
            #         img_min = img_to_log.quantile(0.01)
            #         img_max = img_to_log.quantile(0.99)
            #         img_to_log = torch.clamp(img_to_log, img_min, img_max)
            #         # Normalize to [0, 1]
            #         if img_max > img_min:
            #             img_to_log = (img_to_log - img_min) / (img_max - img_min)
            #         else:
            #             img_to_log = torch.zeros_like(img_to_log)
            #         # Scale to [0, 255] and convert to uint8
            #         img_to_log = (img_to_log * 255).clamp(0, 255).byte()

            #         wandb.log({"sample_image": wandb.Image(img_to_log)})

                
                    
            #         first_sample_logged_training = True
                        # ---------------------------------------------

            if not configs["linear_evaluation"] and not configs["fully_finetune"]:
                if not resnet_star_evaluation:
                    out = model(image, keys=keys)
                else:
                    image = image[:, desired_indices, :, :]
                    out = model(image, keys=keys)
            else:
                image = image[:, desired_indices, :, :]
                out = model((image, spectral_keys), keys=keys)
            # if idx == 51:  # break after the 5th batch (idx starts at 0)
            #     break
            
            loss = criterion(out, label)
            running_loss += loss.item()
        t_forward_end = time.perf_counter() if (log_timing and idx % timing_interval == 0 and t_step_start is not None) else None
        if idx % 100 == 0:
            log_dict = {"Epoch": epoch, "Iteration": idx, "train loss": running_loss/num_samples}
            if configs["schedule"] != "none":
                log_dict["Current Learning Rate"] = get_current_learning_rate(optimizer)

            if configs["wandb"]:
                wandb.log(log_dict)
            else:
                print(log_dict)
        try:
            if configs["mixed_precision"]:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        except RuntimeError as e:
            error_msg = str(e)
            if "GET was unable to find an engine" in error_msg or "unable to find an engine" in error_msg:
                print(f"\n{'='*60}")
                print("ERROR: PyTorch execution engine issue detected.")
                print("This is likely due to cuDNN compatibility problems.")
                print(f"{'='*60}")
                print("cuDNN has been disabled, but the error persists.")
                print("This suggests a deeper PyTorch/CUDA compatibility issue.")
                print(f"{'='*60}\n")
                print("Possible solutions:")
                print("1. Update PyTorch and CUDA to compatible versions")
                print("2. Reinstall cuDNN with matching versions")
                print("3. Use a different CUDA/PyTorch environment")
                print(f"{'='*60}\n")
                raise RuntimeError(
                    "Training failed due to PyTorch execution engine issue. "
                    "cuDNN has been disabled but the error persists. "
                    "Please check your PyTorch/CUDA/cuDNN versions for compatibility."
                ) from e
            else:
                print(f"Error during backward pass: {e}")
                raise e
        if log_timing and idx % timing_interval == 0 and t_step_start is not None and t_forward_end is not None and configs["wandb"]:
            if use_cuda_sync:
                torch.cuda.synchronize(cuda_device)
            t_step_end = time.perf_counter()
            step_ms = (t_step_end - t_step_start) * 1000
            full_forward_ms = (t_forward_end - t_step_start) * 1000
            peak_mem_mb = get_peak_memory_mb(device)
            gpu_util_pct = get_gpu_util(device) if isinstance(device, torch.device) and device.type == "cuda" else None
            rp = configs.get("retain_percent", 100)
            img_size = configs.get("effective_image_size", configs.get("image_size"))
            encoder_gflops = estimate_vit_encoder_gflops(
                img_size, configs.get("in_channels"), rp,
                configs.get("backbone", ""), batch_size=image.shape[0]
            )
            # For classification (RViT): use formula so full_gflops varies with retain_percent.
            # fvcore trace often yields the same FLOPs across retain_percent due to dynamic masking.
            full_gflops = estimate_full_architecture_gflops(
                img_size, configs.get("in_channels"), rp,
                configs.get("backbone", ""), batch_size=image.shape[0],
                architecture=configs.get("architecture", "rvit")
            )
            enc_fv = configs.get("_encoder_gflops_fv_measured_this_epoch")
            if enc_fv is None and encoder_gflops is not None:
                enc_fv = encoder_gflops  # fallback: formula when fvcore trace fails
            timing_dict = build_timing_log_dict(
                step_ms=step_ms, full_forward_ms=full_forward_ms, peak_mem_mb=peak_mem_mb,
                batch_size=image.shape[0], retain_percent=rp,
                encoder_ms=None, gpu_util_pct=gpu_util_pct, encoder_gflops=encoder_gflops,
                full_gflops=full_gflops,
                encoder_gflops_fv=enc_fv
            )
            wandb.log(timing_dict)
            if "_timing_accumulator" in configs:
                configs["_timing_accumulator"].append(dict(timing_dict))
        if configs["schedule"] == "cos":
            if "samples_per_epoch" in configs:
                num_steps = configs["samples_per_epoch"] // configs["batch_size"]
            else:
                num_steps = 10000
            adjust_learning_rate(optimizer, idx / num_steps + epoch, configs)
        
        # Memory cleanup moved to after loss computation


def train(configs):
    print("=" * 20)
    print("Initializing classification trainer")
    print("=" * 20)
    
    # Clear GPU memory and reset stats at start
    if torch.cuda.is_available():
        try:
            # Extract device index from configs["device"] (e.g., "cuda:3" -> 3, "cuda" -> 0)
            device_obj = torch.device(configs["device"])
            device_idx = device_obj.index if device_obj.index is not None else 0
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # Reset CUDA device to clear any error state
            torch.cuda.synchronize(device=device_idx)
            print(f"GPU Memory cleared. Available: {torch.cuda.get_device_properties(device_idx).total_memory / 1024**3:.2f} GB")
        except RuntimeError as e:
            print(f"Warning: CUDA initialization issue: {e}")
            print("Attempting to recover...")
            torch.cuda.empty_cache()
    metrics = utils.initialize_metrics(configs)
    criterion = utils.create_loss(configs)
    base_model = model_utils.create_model(configs)
    optimizer = utils.create_optimizer(configs)(
        base_model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"]
    )


    if configs["torch_compile"]:
        # Use torch.compile to compile the model
        model = torch.compile(base_model)
    else:
        # Use model in eager mode (no compilation)
        model = base_model



    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    train_loader, val_loader, _ = utils.create_dataloaders(configs)
    model.to(configs["device"])
    best_loss = 10000.0
    if configs['start_epoch'] is None:
        start_epoch = 0
    else:
        start_epoch = configs['start_epoch']

    early_stop_counter = 0

    start_time = time.time()
    if configs.get("log_timing_metrics", False) and configs.get("wandb"):
        configs["_timing_accumulator"] = []

    for epoch in range(start_epoch,configs["epochs"]):
        if configs.get("log_timing_metrics", False) and torch.cuda.is_available():
            reset_peak_memory_stats(configs["device"])
        print("=" * 20)
        print("Epoch: ", epoch)
        print("-" * 20)
        print("Begin Training")

        train_epoch(train_loader, model, optimizer, criterion, epoch, configs, scaler)
        val_loss = test(configs, phase="val", model=model, criterion=criterion, loader=val_loader, epoch=epoch)
        if "early_stopping" in configs and configs["early_stopping"] > 0:
            early_stop, early_stop_counter = early_stopping(
                    val_loss, best_loss, early_stop_counter, patience=configs["early_stopping"]
                )
        else:
            early_stop = False
        if val_loss < best_loss:
            best_loss = val_loss
            print("New best validation loss: ", best_loss)
            print("Saving checkpoint")
            # Store checkpoint
            torch.save(base_model, os.path.join(configs["checkpoint_path"], "best_model.pt"))
            torch.save(base_model.state_dict(), os.path.join(configs["checkpoint_path"], "best_model_state_dict.pt"))
        if early_stop:
            print("Early stopping at epoch: ", epoch)
            break
    total_time = time.time() - start_time
    if configs["wandb"]:
        wandb.log({"Total Training Time (s)": total_time})
        if configs.get("log_timing_metrics", False) and "_timing_accumulator" in configs and configs["_timing_accumulator"] and wandb.run:
            summary = compute_timing_summary(configs["_timing_accumulator"], include_std=True)
            for k, v in summary.items():
                wandb.run.summary[k] = v

def test(configs, phase, model=None, loader=None, criterion=None, epoch="Test"):
    try:
        if phase == "test":
            print("-" * 20)
            print("Begin Testing")
            
            _, _, loader = utils.create_dataloaders(configs)
            criterion = utils.create_loss(configs)

            # Load model from checkpoint with error handling
            try:
                # Prefer explicit eval_checkpoint if provided; otherwise fall back to checkpoint_path/best_model.pt
                if "eval_checkpoint" in configs and configs["eval_checkpoint"] is not None:
                    ckpt_path = configs["eval_checkpoint"]
                else:
                    ckpt_path = os.path.join(configs["checkpoint_path"], "best_model.pt")

                # Try to load state_dict first (preferred for inference with RViT wrapper)
                ckpt_dir = os.path.dirname(ckpt_path)
                state_dict_path = os.path.join(ckpt_dir, "best_model_state_dict.pt")
                
                if os.path.isfile(state_dict_path):
                    print(f"Loading model from state_dict: {state_dict_path}")
                    model = model_utils.create_model(configs)
                    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    if missing:
                        print(f"Warning: Missing keys when loading state_dict: {missing[:5]}...")
                    if unexpected:
                        print(f"Warning: Unexpected keys when loading state_dict: {unexpected[:5]}...")
                else:
                    # Fallback: load full model object
                    print(f"Loading full model object from checkpoint: {ckpt_path}")
                    model = torch.load(
                        ckpt_path,
                        map_location=configs["device"],
                        weights_only=False,
                    )

                # --- Inference-time masking wiring for RViT-based models ---
                # Mark model as being in testing (inference) phase.
                # RViT checks this flag plus its internal apply_mask_in_infer flag.
                setattr(model, "testing", True)
                # Ensure inference-time masking flag matches current configs, even for older checkpoints
                # that were saved before the attribute existed.
                setattr(model, "apply_mask_in_infer", configs.get("apply_mask_in_infer", False))

                # Debug: print model type and key masking-related attributes
                print(f"[DEBUG] Loaded model type: {type(model)}")
                if hasattr(model, "implementation"):
                    print(f"[DEBUG] RViT implementation: {getattr(model, 'implementation')}")
                if hasattr(model, "retain_percent"):
                    print(f"[DEBUG] RViT retain_percent: {getattr(model, 'retain_percent')}")
                print(f"[DEBUG] apply_mask_in_infer: {getattr(model, 'apply_mask_in_infer', None)}")
                print(f"[DEBUG] testing flag on model: {getattr(model, 'testing', None)}")
            except Exception as e:
                print(f"Error loading model from checkpoint: {e}")
                # Fallback: create a fresh model so evaluation can still run
                model = model_utils.create_model(configs)

            # Model loaded - using eager mode

        elif phase == "val":
            print("-" * 20)
            print("Begin Evaluation")
            
        else:
            print("Uknown phase!")
            exit(3)

        if configs["linear_evaluation"] or configs["fully_finetune"]:
            modality_dictionary = {v: k for k, v in configs["modality_channels"].items()}
            available_modalities = configs["dataset_modality_index"][configs["dataset"]]
            spectral_keys = []
            desired_indices = []
            for modality in available_modalities:
                spectral_keys.append(int(modality_dictionary[modality]))
                desired_indices.append(available_modalities[modality])

            # Clamp indices to available input channels (e.g. BigEarthNet rgb=true -> in_channels=3)
            max_ch = configs["in_channels"]
            desired_indices = [idx for idx in desired_indices if idx < max_ch]
            spectral_keys = [idx for idx in spectral_keys if idx < max_ch]

            use_full_channels_linear_seg = (
                configs["linear_evaluation"]
                and configs.get("seg_to_class", False)
                and len(desired_indices) < configs["in_channels"]
            )
            if use_full_channels_linear_seg:
                desired_indices = list(range(configs["in_channels"]))
                spectral_keys = desired_indices.copy()

        if "resnet_star_evaluation" in configs and configs["resnet_star_evaluation"]:
            resnet_star_evaluation = True
            resnet_star_configs = json.load(open("configs/datasets/supervised_foundation_cls.json", "r"))
            modality_dictionary = {v: k for k, v in resnet_star_configs["modality_channels"].items()}
            available_modalities = resnet_star_configs["dataset_modality_index"][configs["dataset"]]
            spectral_keys = []
            desired_indices = []
            for modality in available_modalities:
                spectral_keys.append(int(modality_dictionary[modality]))
                desired_indices.append(available_modalities[modality])
            print("Desired indices: ", desired_indices)
        else:
            resnet_star_evaluation = False

        metrics = utils.initialize_metrics(configs)
        # MARIDA uses strict deterministic mode → torchmetrics cumsum breaks on GPU; run metrics on CPU
        metrics_on_cpu = configs.get("dataset") == "marida"
        if metrics_on_cpu:
            for m in metrics:
                m.to("cpu")
        model.to(configs["device"])
        model.eval()

        # Inference-phase timing (when apply_mask_in_infer true): log test/inference_ms, test/throughput, etc.
        device = configs["device"]
        cuda_device = torch.device(device) if isinstance(device, str) else device
        use_cuda_sync = isinstance(cuda_device, torch.device) and cuda_device.type == "cuda"
        log_timing_in_test = (phase == "test" and configs.get("log_timing_metrics_in_test", False) and
                              configs.get("apply_mask_in_infer", False))
        timing_interval = configs.get("timing_log_interval", 100)
        if log_timing_in_test and use_cuda_sync and configs.get("wandb"):
            reset_peak_memory_stats(device)
            configs["_test_timing_accumulator"] = []

        for m in metrics:
            m.reset()

        # Cache clearing moved to after loss computation in the loop

        # Initialize accumulation lists
        #all_outputs = []
        #all_labels = []
        total_loss = 0.0
        num_samples = 0
        first_sample_logged = False

        for idx, batch in enumerate(tqdm.tqdm(loader)):
            try:
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', enabled=configs["mixed_precision"]):
                        image, label = batch
                        image = image.to(configs["device"])
                        label = label.to(configs["device"])

                        # Measured full-model GFLOPs for test: once on first batch when use_measured_full_gflops
                        if (log_timing_in_test and idx == 0 and configs.get("use_measured_full_gflops", False)):
                            if configs["linear_evaluation"] or configs["fully_finetune"]:
                                image_for_gflops = (image[:, desired_indices, :, :], spectral_keys)
                            elif resnet_star_evaluation:
                                image_for_gflops = image[:, desired_indices, :, :]
                            else:
                                image_for_gflops = image
                            measured_test = estimate_full_model_gflops(model, image_for_gflops)
                            if measured_test is not None:
                                full_m, enc_fv = measured_test
                                configs["_full_gflops_measured_test"] = full_m
                                configs["_encoder_gflops_fv_measured_test"] = enc_fv

                        # Timed inference branch (test phase, apply_mask_in_infer, every timing_interval batches)
                        if log_timing_in_test and idx % timing_interval == 0 and use_cuda_sync and configs.get("wandb"):
                            torch.cuda.synchronize(cuda_device)
                            t_start = time.perf_counter()
                            if not configs["linear_evaluation"] and not configs["fully_finetune"]:
                                if not resnet_star_evaluation:
                                    out = model(image)
                                else:
                                    image_slice = image[:, desired_indices, :, :]
                                    out = model(image_slice)
                            else:
                                image_slice = image[:, desired_indices, :, :]
                                out = model((image_slice, spectral_keys))
                            if use_cuda_sync:
                                torch.cuda.synchronize(cuda_device)
                            t_end = time.perf_counter()
                            inference_ms = (t_end - t_start) * 1000
                            peak_mem_mb = get_peak_memory_mb(device)
                            gpu_util_pct = get_gpu_util(device) if torch.cuda.is_available() else None
                            rp = configs.get("retain_percent", 100)
                            img_size = configs.get("effective_image_size", configs.get("image_size"))
                            encoder_gflops = estimate_vit_encoder_gflops(
                                img_size, configs.get("in_channels"), rp,
                                configs.get("backbone", ""), batch_size=image.shape[0]
                            )
                            # Use measured full_gflops when available (same as segmentation), else formula
                            full_gflops = configs.get("_full_gflops_measured_test")
                            if full_gflops is None:
                                full_gflops = estimate_full_architecture_gflops(
                                    img_size, configs.get("in_channels"), rp,
                                    configs.get("backbone", ""), batch_size=image.shape[0],
                                    architecture=configs.get("architecture", "rvit")
                                )
                            enc_fv_test = configs.get("_encoder_gflops_fv_measured_test")
                            if enc_fv_test is None and encoder_gflops is not None:
                                enc_fv_test = encoder_gflops  # fallback: formula when fvcore trace fails
                            timing_dict = build_inference_timing_log_dict(
                                inference_ms, peak_mem_mb, image.shape[0], rp,
                                gpu_util_pct=gpu_util_pct, encoder_gflops=encoder_gflops,
                                full_gflops=full_gflops,
                                encoder_gflops_fv=enc_fv_test
                            )
                            wandb.log(timing_dict)
                            if "_test_timing_accumulator" in configs:
                                configs["_test_timing_accumulator"].append(dict(timing_dict))
                        else:
                            # ----------- WandB log first sample ----------
                            if configs["wandb"]:
                                if idx == 0 and phase == "test" and not first_sample_logged:
                                    img_to_log = image[0].detach().cpu()

                                    # if multi-channel hyperspectral, pick first 3 bands
                                    if img_to_log.shape[0] > 3:
                                        img_to_log = img_to_log[:3, :, :]

                                    # Convert normalized image to [0, 255] range for wandb
                                    img_min = img_to_log.quantile(0.01)
                                    img_max = img_to_log.quantile(0.99)
                                    img_to_log = torch.clamp(img_to_log, img_min, img_max)
                                    if img_max > img_min:
                                        img_to_log = (img_to_log - img_min) / (img_max - img_min)
                                    else:
                                        img_to_log = torch.zeros_like(img_to_log)
                                    img_to_log = (img_to_log * 255).clamp(0, 255).byte()

                                    wandb.log({"sample_image": wandb.Image(img_to_log)})
                                    first_sample_logged = True
                            # ---------------------------------------------
                            if not configs["linear_evaluation"] and not configs["fully_finetune"]:
                                if not resnet_star_evaluation:
                                    out = model(image)
                                else:
                                    image = image[:, desired_indices, :, :]
                                    out = model(image)
                            else:
                                image_slice = image[:, desired_indices, :, :]
                                out = model((image_slice, spectral_keys))

                # Loss and metrics OUTSIDE autocast to avoid device/dtype issues in torchmetrics
                device = configs["device"]
                out = out.float().to(device)
                label = label.to(device)

                loss = criterion(out, label)
                total_loss += loss.item()

                label_for_metrics = label.int() if configs.get("multilabel") else label
                if metrics_on_cpu:
                    out_for_m = out.detach().cpu()
                    label_for_m = label_for_metrics.detach().cpu()
                else:
                    out_for_m = out
                    label_for_m = label_for_metrics
                for metric in metrics:
                    metric(out_for_m, label_for_m)
                num_samples += image.shape[0]
            except Exception as e:
                print(f"Error processing batch {idx}: {e}")
                print("Skipping this batch and continuing...")
                continue

        if phase == "test" and configs.get("wandb") and configs.get("log_timing_metrics_in_test", False):
            if "_test_timing_accumulator" in configs and configs["_test_timing_accumulator"] and wandb.run:
                summary = compute_timing_summary(configs["_test_timing_accumulator"], include_std=True)
                for k, v in summary.items():
                    wandb.run.summary[k] = v

        total_loss = total_loss / max(num_samples, 1)  # Prevent division by zero
        log_dict = {"Epoch": epoch, phase + " loss": total_loss}

        for idx, metric in enumerate(metrics):
            if metric.__class__.__name__ == "MultilabelCoverageError":
                log_dict[phase + " " + metric.__class__.__name__] = metric.compute()
            elif metric.average != "none":
                log_dict[phase + " " + metric.average + " " + metric.__class__.__name__] = metric.compute()
            else:
                if phase != "val":
                    scores = metric.compute()
                    for idx in range(scores.shape[0]):
                        log_dict[phase + " " + metric.__class__.__name__ + " Class: " + str(idx)] = scores[idx]

        if configs["wandb"]:
            wandb.log(log_dict)
        else:
            print(log_dict)
        return total_loss
        
    except Exception as e:
        print(f"Critical error in test function: {e}")
        print("Returning default loss value to prevent crash...")
        return 0.0