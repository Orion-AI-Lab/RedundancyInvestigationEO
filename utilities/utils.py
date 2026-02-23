import glob
import importlib
import io
import os
import pprint
import random
from pathlib import Path

import einops
import numpy as np
import pyjson5 as json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import webdataset as wds
from braceexpand import braceexpand
from torchmetrics import Accuracy, F1Score, JaccardIndex, Precision, Recall, AveragePrecision
from torchmetrics.classification import MultilabelCoverageError
import warnings
import albumentations as A

import datasets

import utilities.augmentations as augmentations
import utilities.webdataset_writer as web_writer


class SafeAveragePrecision(AveragePrecision):
    """Wrapper for AveragePrecision that handles nan values gracefully"""
    
    def compute(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            result = super().compute()
            # Replace nan values with 0.0 for classes with no positive samples
            if torch.is_tensor(result):
                result = torch.nan_to_num(result, nan=0.0)
            return result

def create_checkpoint_path(configs):
    if configs["checkpoint_root_path"] is None:
        checkpoint_root_path = Path("checkpoints")
    else:
        checkpoint_root_path = Path(os.path.expandvars(configs["checkpoint_root_path"]))

    if "fully_finetune" in configs:
        fully_finetune = configs["fully_finetune"]
    else:
        fully_finetune = False

    if configs["linear_evaluation"]:
        finetuning_path = "linear_eval"
        checkpoint_root_path = checkpoint_root_path / finetuning_path
    elif fully_finetune:
        finetuning_path = "fully_finetune"
        checkpoint_root_path = checkpoint_root_path / finetuning_path

    if configs["task"] == "segmentation":
        checkpoint_path = (
            checkpoint_root_path
            / configs["task"].lower()
            / configs["dataset"].lower()
            / configs["architecture"].lower()
            / configs["backbone"].lower()
        )
    elif configs["task"] == "classification":
        checkpoint_path = (
            checkpoint_root_path / configs["task"].lower() / configs["dataset"].lower() / configs["backbone"].lower()
        )
    elif configs["task"] == "mae":
        checkpoint_path = (
            checkpoint_root_path
            / configs["task"].lower()
            / configs["dataset"].lower()
            / (
                "vit_patch_"
                + str(configs["patch_size"])
                + "_depth_"
                + str(configs["depth"])
                + "_heads_"
                + str(configs["heads"])
            )
            / ("mask_ratio_" + str(configs["masked_ratio"]))
        )

        if configs["single_embedding_layer"]:
            checkpoint_path = checkpoint_path / "single_embedding_layer"
        else:
            checkpoint_path = checkpoint_path / "multiple_embedding_layers"
    elif configs["task"] == "supervised_foundation":
        checkpoint_path = (
            checkpoint_root_path / configs["task"].lower() / configs["dataset"].lower() / configs["backbone"].lower()
        )
    else:
        print("Task not supported!")
        exit(3)

    # Add the run name to the checkpoint path
    # Handle case where wandb_run_name might be None (e.g., if wandb.init() hasn't completed)
    run_name = configs.get("wandb_run_name")
    if run_name is None:
        # Use a default name if wandb_run_name is not set
        run_name = "unnamed_run"
    checkpoint_path = checkpoint_path / run_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def load_config_file(path):
    if not os.path.isfile(path):
        print("=" * 20)
        print("Path: ", path, " does not exist.")
        print("Exiting")
        print("=" * 20)
        exit(1)
    data = json.load(open(path, "r"))
    return data


def load_configs(args=None):
    custom_config = False
    custom_training_config = False
    custom_dataset_config = False
    custom_method_config = False
    custom_augmentation_config = False
    wandb_id_resume = False

    if args.config is not None:
        custom_config = True
    if args.training_config is not None:
        custom_training_config = True
    if args.dataset_config is not None:
        custom_dataset_config = True
    if args.method_config is not None:
        custom_method_config = True
    if args.augmentation_config is not None:
        custom_augmentation_config = True
    if args.wandb_id_resume is not None:
        wandb_id_resume = True

    # Load data related configs
    if not custom_config:
        print("No valid config provided")
        exit(1)
    else:
        root_config_path = args.config
    config = json.load(open(root_config_path, "r"))
    #print(config)

    if wandb_id_resume:
        config["wandb_id_resume"] = args.wandb_id_resume

    # Load dataset related configs
    dataset = config["dataset"]
    
    if not custom_dataset_config:
        dataset_config_path = "configs/datasets/" + str(dataset).lower() + ".json"
    else:
        dataset_config_path = args.dataset_config
        config["dataset"] = dataset_config_path.split("/")[-1][:-5]
    dataset_config = load_config_file(dataset_config_path)
    config.update(dataset_config)

    # Load method related configs
    if not custom_method_config:
        #method_config_path = "configs/method/" + config["task"].lower() + ".json"
        print("No valid method config provided")
        exit(1)
    else:
        method_config_path = args.method_config

    method_config = load_config_file(method_config_path)
    if args.mask_threshold is not None:
        method_config["mask_threshold"] = args.mask_threshold
    if args.implementation is not None:
        method_config["implementation"] = args.implementation
    if args.attention_mask is not None:
        method_config["attention_mask"] = args.attention_mask
    if args.backbone is not None:
        method_config["backbone"] = args.backbone
    if args.retain_percent is not None:
        method_config["retain_percent"] = args.retain_percent
    if args.pretrained is not None:
        method_config["pretrained"] = args.pretrained
    if getattr(args, "apply_mask_in_infer", None) is not None:
        method_config["apply_mask_in_infer"] = args.apply_mask_in_infer
    config.update(method_config)

    # Load training configs
    if not custom_training_config:
        #training_configs_path = "configs/training/training.json"
        print("No valid training config provided")
        exit(1)
    else:
        training_configs_path = args.training_config
    training_configs = load_config_file(training_configs_path)
    config.update(training_configs)

    # MARIDA: if marida_merge_debris_subclasses is True, use 8 thematic classes (merge 1–4 → 1)
    # and ignore_index=8; otherwise 11 thematic classes and ignore_index=11.
    if config.get("dataset") == "marida":
        if config.get("marida_merge_debris_subclasses", False):
            config["num_classes"] = 9
            config["ignore_index"] = 8
        else:
            config["num_classes"] = 12
            config["ignore_index"] = 11

    # Load normalization configs
    if config["normalization"] == "standard":
        stats_config_path = "configs/stats/stats.json"
        stats_config = load_config_file(stats_config_path)
        config["mean"] = stats_config[config["dataset"]]["mean"]
        config["std"] = stats_config[config["dataset"]]["std"]

    # Load augmentation configurations
    if config["augment"]:
        augmentation_configs_path = args.augmentation_config or config.get("augmentation_config")
        if augmentation_configs_path is not None:
            augmentations_config = load_config_file(augmentation_configs_path)
            config.update(augmentations_config)
        else:
            print("Warning: augment=True but no augmentation_config provided. Using default augmentations.")
            config["augmentations"] = {}
    else:
        config["augmentations"] = {}
    if config["device"] == "cpu":
        print("Disabling mixed precision. Incompatible with CPU.")
        config["mixed_precision"] = False

    if "multilabel" not in config:
        config["multilabel"] = False

    if config["distributed"]:
        distributed_config_path = os.path.join("configs", "distributed", "distributed.json")
        distributed_config = load_config_file(distributed_config_path)
        config.update(distributed_config)

    if "fully_finetune" in config:
        fully_finetune = config["fully_finetune"]
    else:
        fully_finetune = False
        config["fully_finetune"] = fully_finetune

    if config["linear_evaluation"] or fully_finetune:
        multimodal_mae_config_path = os.path.join("configs", "datasets", "all.json")
        multimodal_mae_config = load_config_file(multimodal_mae_config_path)
        mae_config = {
            "modality_channels": dict(multimodal_mae_config["modality_channels"]),
            "dataset_modality_index": dict(multimodal_mae_config["dataset_modality_index"]),
        }
        # Merge in current dataset's modality config (e.g. mlrsnet) so it is not overwritten
        if "modality_channels" in config:
            mae_config["modality_channels"].update(config["modality_channels"])
        if "dataset_modality_index" in config:
            mae_config["dataset_modality_index"].update(config["dataset_modality_index"])
        config.update(mae_config)

    # When rgb=true, dataloader slices to 3 channels (e.g. BigEarthNet B04,B03,B02). Model must expect 3 channels.
    if config.get("rgb") and config.get("dataset") in ("bigearthnet", "sen12ms"):
        config["in_channels"] = 3

    return config


def get_dataset(dataset_name, configs, mode):
    if dataset_name.lower() == "flair":
        dataset = datasets.FLAIRDataset.FLAIRDataset(configs, mode)
    elif dataset_name.lower() == "flair2":
        dataset = datasets.FLAIR2Dataset.FLAIR2Dataset(configs, mode)
    elif dataset_name.lower() == "woody":
        dataset = datasets.WoodyDataset.WoodyDataset(configs, mode)
    elif dataset_name.lower() == "waititu":
        dataset = datasets.WaitituDataset.WaitituDataset(configs, mode)
    elif dataset_name.lower() == "mlrsnet":
        dataset = datasets.MLRSNetDataset.MLRSNetDataset(configs, mode)
    elif dataset_name.lower() == "bigearthnet":
        dataset = datasets.BigEarthNetDataset.BigEarthNetDataset(configs, mode)
    elif dataset_name.lower() == "marida":
        dataset = datasets.MaridaDataset.GenDEBRIS(configs, dataset_name, mode)
    else:
        print("Dataset not supported")
        exit(2)
    return dataset


def load_dataset(configs, mode):
    # import all dataset definitions
    module_files = [f for f in os.listdir("datasets") if f.endswith(".py")]
    for module_file in module_files:
        module_name = module_file[:-3]
        importlib.import_module(f"datasets.{module_name}")
    if configs["dataset"] == "all":
        datasets_in_consideration = configs["train_datasets"]
        all_datasets = []
        for idx, dataset in enumerate(datasets_in_consideration):
            all_datasets.append(dataset)
        # dataset = UniversalDataset.UniversalDataset(all_datasets,configs,mode)
        print("UniversalDataloader is supported only with webdataset option on! Exiting!")
        exit(3)
    else:
        dataset = get_dataset(configs["dataset"], configs, mode)
    return dataset


def create_dataloaders(configs):
    if "webdataset" in configs:
        if configs["webdataset"]:
            if configs["dataset"] != "all":
                return create_webdataset_loaders(configs)
            else:
                return create_universal_webdataset_loader(configs)

    train_dataset = load_dataset(configs, "train")
    val_dataset = load_dataset(configs, "val")

    test_dataset = load_dataset(configs, "test")

    if hasattr(train_dataset, "collate_fn") and callable(getattr(train_dataset, "collate_fn")):
        collate_function = train_dataset.collate_fn
    else:
        collate_function = None

    drop_last_eval = False
    if "architecture" in configs:
        if configs["architecture"] == "fasterrcnn" and (configs["linear_evaluation"] or
                                                        configs["backbone"] == "dinov2"):
            drop_last_eval = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True,
        num_workers=configs["num_workers"],
        pin_memory=configs["pin_memory"],
        drop_last=True,
        collate_fn=collate_function,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs["batch_size"],
        shuffle=False,
        num_workers=configs["num_workers"],
        pin_memory=configs["pin_memory"],
        drop_last=drop_last_eval,
        collate_fn=collate_function,
    )
    if isinstance(test_dataset, list):
        # several dataset to test on separately
        test_loader = [torch.utils.data.DataLoader(
            sub_test_dataset,
            batch_size=configs["batch_size"],
            shuffle=False,
            num_workers=configs["num_workers"],
            pin_memory=configs["pin_memory"],
            drop_last=drop_last_eval,
            collate_fn=collate_function,
        )
                       for sub_test_dataset in test_dataset]
    else:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=configs["batch_size"],
            shuffle=False,
            num_workers=configs["num_workers"],
            pin_memory=configs["pin_memory"],
            drop_last=drop_last_eval,
            collate_fn=collate_function,
        )
    return train_loader, val_loader, test_loader


def create_webdataset_loaders(
    configs, repeat=False, self_supervised=False, resample_shards=False, yield_dataset=False, supervised_foundation=False
):
    def get_patches(src):
        for sample in src:
            image = torch.load(io.BytesIO(sample["image.pth"])).float()
            label = torch.load(io.BytesIO(sample["labels.pth"]))
            if configs['dataset']=='marida':
                # MaridaDataset saves images in CHW format [C, H, W] after the fix
                # But old webdatasets might have HWC format, so check and rearrange if needed
                if len(image.shape) == 3:
                    # Check if it's HWC (last dim matches in_channels) or CHW (first dim matches in_channels)
                    if image.shape[-1] == configs['in_channels'] and image.shape[0] != configs['in_channels']:
                        # It's HWC [H, W, C], rearrange to CHW [C, H, W]
                        image = einops.rearrange(image, "h w c-> c h w")
                    # Ensure it's CHW format [C, H, W] for normalization
                    assert image.shape[0] == configs['in_channels'], f"Marida image should be CHW [C={configs['in_channels']}, H, W], got shape {image.shape}"
                # Handle label remapping: no-data -1 -> 11 (unlabeled)
                if isinstance(label, torch.Tensor):
                    label[label == -1] = 11
                elif isinstance(label, (int, np.integer)) and label == -1:
                    label = 11

                # MARIDA: merge debris subclasses 1–4 when marida_merge_debris_subclasses is True (8 thematic + ignore 8).
                if configs.get("marida_merge_debris_subclasses", False) and isinstance(label, torch.Tensor):
                    t = label.clone()
                    mask_1_4 = (t >= 1) & (t <= 4)
                    t[mask_1_4] = 1
                    for old, new in [(5, 2), (6, 3), (7, 4), (8, 5), (9, 6), (10, 7)]:
                        t[t == old] = new
                    t[t == 11] = 8
                    label = t

                # Optional MARIDA patch-level filtering: keep only tiles whose fraction of
                # non-11 patches is above a user-defined threshold (0-100).
                thr = configs.get("marida_non11_patch_threshold", None)
                if thr is not None and isinstance(label, torch.Tensor):
                    patch_size = configs.get("finetuning_patch_size", 16)
                    lbl = label
                    if lbl.dim() == 3:
                        # Some MARIDA labels might have shape [1, H, W]
                        lbl = lbl.squeeze(0)
                    if lbl.dim() != 2:
                        raise ValueError(f"Expected MARIDA label to be 2D [H,W] or [1,H,W], got shape {label.shape}")
                    h, w = lbl.shape
                    pad_h = (patch_size - h % patch_size) % patch_size
                    pad_w = (patch_size - w % patch_size) % patch_size
                    # use current ignore_index (default 11 for MARIDA paper setup)
                    ignore_label = configs.get("ignore_index", 11)
                    lbl_padded = F.pad(lbl, (0, pad_w, 0, pad_h), value=ignore_label)
                    Hp, Wp = lbl_padded.shape
                    n_h = Hp // patch_size
                    n_w = Wp // patch_size
                    patches = lbl_padded.view(n_h, patch_size, n_w, patch_size).permute(0, 2, 1, 3)
                    non11_patch_mask = (patches != ignore_label).any(dim=(-1, -2))
                    num_non11_patches = non11_patch_mask.sum().item()
                    total_patches = non11_patch_mask.numel()
                    ratio = (num_non11_patches / total_patches * 100.0) if total_patches > 0 else 0.0
                    if ratio < float(thr):
                        # Skip this sample entirely
                        continue
            if configs['dataset']=='mlrsnet':
                    label = label.float()
            if configs['dataset']=='sen12ms':
                label[label>=configs['threshold']] = 1
                label[label<configs['threshold']] = 0                
                    
            '''if configs['dataset'] == 'flair2':
                print(image.shape)'''
            if configs['dataset']=='tallos':
                image[image>3e38] = 0.0
                image = torch.nan_to_num(image,nan=0.0)            
                image[torch.isinf(image)] = 0.0
                
            if configs['dataset'] == 'treesatai' and 's1_60m' in configs['data_source']:
                image = image[:2, :, :]

            if isinstance(label, dict):
                label = label["label"]
                if configs["dataset"] == "rapidai4eo":
                    label[label >= configs["class_threshold"]] = 1
                    label[label < configs["class_threshold"]] = 0
               
            if configs["dataset"] == "uav" or configs["dataset"] == "woody":
                # Remove alpha channel from combined uav datasets (Some have it some don't)
                image = image[:3, :, :]
                if not configs["dataset"] == "woody":
                    label = torch.zeros(1)
            if configs["dataset"] == "waititu" and image.shape[0] > configs["in_channels"]:
                # Slice to in_channels only (e.g. 4 for full Waititu, or 3 for RGB-only)
                image = image[: configs["in_channels"], :, :]
            if len(image.shape) > 3:
                if not configs["timeseries"]:
                    random_index = random.randint(0, 3)
                    image = image[random_index]

            if configs["enforce_resize"] is not None:
                if (
                    configs["linear_evaluation"]
                    or configs["fully_finetune"]
                    or configs["dataset"] == "treesatai"
                    or configs["dataset"] == "forestnet"
                    or configs["dataset"] == "fivebillionpixels"
                    or configs["dataset"] == "flair"
                    or configs["dataset"] == "flair2"
                    or configs["dataset"] == "bigearthnet"
                ):
                    if configs["task"] == "classification":
                        image = image.permute(1, 2, 0).numpy()
                        aug = A.Compose(
                            [
                                A.augmentations.Resize(
                                    height=configs["enforce_resize"],
                                    width=configs["enforce_resize"],
                                    p=1.0,
                                )
                            ]
                        )(image=image)
                        image = aug["image"]
                        image = einops.rearrange(image, "h w c -> c h w")
                        image = torch.from_numpy(image).float()
                    elif configs["task"] == "segmentation":
                        image = image.permute(1, 2, 0).numpy()
                        aug = A.Compose(
                            [
                                A.augmentations.Resize(
                                    height=configs["enforce_resize"],
                                    width=configs["enforce_resize"],
                                    p=1.0,
                                )
                            ]
                        )(image=image, mask=label.numpy())
                        image = aug["image"]
                        label = aug["mask"]
                        image = einops.rearrange(image, "h w c -> c h w")
                        image = torch.from_numpy(image).float()
                        label = torch.from_numpy(label).long()
            if configs["augment"]:
                data_augmentations = augmentations.get_augmentations(configs)
                if configs["task"] == "classification" or configs["task"] == "mae":
                    image = image.permute(1, 2, 0).numpy()
                    transform = data_augmentations(image=image)
                    image = transform["image"]
                    image = einops.rearrange(image, "h w c -> c h w")
                    image = torch.from_numpy(image).float()
                elif configs["task"] == "segmentation":
                    image = image.permute(1, 2, 0).numpy()
                    transform = data_augmentations(image=image, mask=label.numpy())
                    image = transform["image"]
                    image = einops.rearrange(image, "h w c -> c h w")
                    label = transform["mask"]
                    label = torch.from_numpy(label)
                    image = torch.from_numpy(image).float()
                else:
                    print("Augmentations not supported for this task. Continuing without augmentations!")
            if configs["normalization"] == "minmax":
                if configs["dataset"] == "sen12ms":
                    s1 = image[:2, :, :] / 25 + 1
                    s2 = image[2:, :, :] / 10000
                    image = torch.cat((s1, s2), dim=0)
                else:
                    image /= image.max() + 1e-6
            elif configs["normalization"] == "standard":
                if "mean" not in configs or "std" not in configs:
                    print("Mean and Std not provided for this dataset. Exiting!")
                    exit(2)
                if configs['dataset'] == 'bigearthnet':
                    if configs['rgb']:
                        # Channel order: VV,VH,B01,B02,B03,B04,... → RGB = B04,B03,B02 = indices 5,4,3
                        b = image[3,:,:]
                        g = image[4,:,:]
                        r = image[5,:,:]
                        image = torch.stack((r,g,b), dim=0)
                        mean = list(reversed(configs["mean"][3:6]))
                        std = list(reversed(configs["std"][3:6]))
                        normalization = transforms.Normalize(mean=mean, std=std)
                if configs["dataset"] == "sen12ms":
                    if configs['rgb']:
                        r = image[5,:,:]
                        g = image[4,:,:]
                        b = image[3,:,:]
                        image = torch.stack((r,g,b), dim=0)
                        mean = list(reversed(configs["mean"][3:6]))
                        std = list(reversed(configs["std"][3:6]))
                        normalization = transforms.Normalize(mean=mean, std=std)
                    elif configs['sar']:
                        image = image[:2,:,:]
                        mean = list((configs["mean"][:2]))
                        std = list((configs["std"][:2]))
                        normalization = transforms.Normalize(mean=mean, std=std)
                if "rgb" not in configs or (not configs['rgb'] and ('sar' not in configs or not configs['sar'])):
                    normalization = transforms.Normalize(mean=configs["mean"], std=configs["std"])
                image = normalization(image)
                
            if configs['dataset']=='bigearthnet' and configs['num_classes']==5:
                inverted_nomenclature_five = {2: 0, 14: 0, 0: 1, 16: 1, 3: 1, 6: 1, 15: 1, 13: 1, 17: 2, 10: 2, 12: 2, 1: 2, 5: 2, 7: 2, 18: 2, 8: 3, 9: 3, 4: 4, 11: 4}
                l5 = torch.zeros(5)
                for i,_ in enumerate(label):
                    l5[inverted_nomenclature_five[i]] = l5[inverted_nomenclature_five[i]] or label[i]
                label = l5
            
            
            #Add some checks
            if configs['dataset']=="flair" and configs['multilabel']:
                assert configs['task']=="classification", "Multilabel supported only for classification task in flair dataset"
            if configs['dataset']=="flair" and configs['multilabel'] and configs['task']=="classification":
                num_classes = configs['num_classes']
                unique_labels = torch.unique(label)
                multilabel_target = torch.zeros(num_classes, dtype=torch.float)
                multilabel_target[unique_labels] = 1
                label = multilabel_target
            
            if not yield_dataset:
                yield (image, label, sample["__key__"])
            else:
                yield (image, label, sample["__key__"], configs["dataset"])

    def get_patches_eval(src):
        for sample in src:
            image = torch.load(io.BytesIO(sample["image.pth"])).float()
            label = torch.load(io.BytesIO(sample["labels.pth"]))
            if configs['dataset']=='marida':
                # Ensure MARIDA images are in CHW format [C, H, W], matching training path logic.
                if len(image.shape) == 3:
                    # If last dim matches in_channels and first doesn't, it's HWC -> convert to CHW
                    if image.shape[-1] == configs['in_channels'] and image.shape[0] != configs['in_channels']:
                        image = einops.rearrange(image, "h w c-> c h w")
                    # Final safety check
                    assert image.shape[0] == configs['in_channels'], (
                        f"Marida image should be CHW [C={configs['in_channels']}, H, W], "
                        f"got shape {image.shape}"
                    )
                # Handle label remapping: no-data -1 -> 11
                if isinstance(label, torch.Tensor):
                    label[label == -1] = 11
                elif isinstance(label, (int, np.integer)) and label == -1:
                    label = 11

                # MARIDA: merge debris subclasses 1–4 when marida_merge_debris_subclasses is True (8 thematic + ignore 8).
                if configs.get("marida_merge_debris_subclasses", False) and isinstance(label, torch.Tensor):
                    t = label.clone()
                    mask_1_4 = (t >= 1) & (t <= 4)
                    t[mask_1_4] = 1
                    for old, new in [(5, 2), (6, 3), (7, 4), (8, 5), (9, 6), (10, 7)]:
                        t[t == old] = new
                    t[t == 11] = 8
                    label = t
            if configs['dataset']=='mlrsnet':
                    label = label.float()
              
            if isinstance(label, dict):
                label = label["label"]

                if configs['dataset']=='mlrsnet':
                    labels = labels.float()
            if configs["enforce_resize"] is not None:
                if (
                    configs["linear_evaluation"]
                    or configs["fully_finetune"]
                    or configs["dataset"] == "treesatai"
                    or configs["dataset"] == "forestnet"
                    or configs["dataset"] == "fivebillionpixels"
                    or configs["dataset"] == "flair"
                    or configs["dataset"] == "flair2"
                    or configs["dataset"] == "bigearthnet"
                ):
                    if configs["task"] == "classification":
                        image = image.permute(1, 2, 0).numpy()
                        aug = A.Compose([
                            A.augmentations.Resize(
                                height=configs["enforce_resize"],
                                width=configs["enforce_resize"],
                                p=1.0,
                            )
                        ])(image=image)
                        image = aug["image"]
                        image = einops.rearrange(image, "h w c -> c h w")
                        image = torch.from_numpy(image).float()
                    elif configs["task"] == "segmentation":
                        image = image.permute(1, 2, 0).numpy()
                        aug = A.Compose(
                            [
                                A.augmentations.Resize(
                                    height=configs["enforce_resize"],
                                    width=configs["enforce_resize"],
                                    p=1.0,
                                )
                            ]
                        )(image=image, mask=label.numpy())
                        image = aug["image"]
                        label = aug["mask"]
                        image = einops.rearrange(image, "h w c -> c h w")
                        image = torch.from_numpy(image).float()
                        label = torch.from_numpy(label).long()

            if configs["dataset"] == "woody":
                # Remove alpha channel from combined uav datasets (Some have it some don't)
                image = image[:3, :, :]
            if configs["dataset"] == "waititu" and image.shape[0] > configs["in_channels"]:
                image = image[: configs["in_channels"], :, :]  # match dataset in_channels
            if configs["normalization"] == "minmax":
                    image /= image.max() + 1e-6
            elif configs["normalization"] == "standard":
                if "mean" not in configs or "std" not in configs:
                    print("Mean and Std not provided for this dataset. Exiting!")
                    exit(2)
                if configs['dataset'] == 'bigearthnet':
                    if configs['rgb']:
                        # Channel order: VV,VH,B01,B02,B03,B04,... → RGB = B04,B03,B02 = indices 5,4,3
                        b = image[3,:,:]
                        g = image[4,:,:]
                        r = image[5,:,:]
                        image = torch.stack((r,g,b), dim=0)
                        mean = list(reversed(configs["mean"][3:6]))
                        std = list(reversed(configs["std"][3:6]))
                        normalization = transforms.Normalize(mean=mean, std=std)
                if "rgb" not in configs or (not configs['rgb'] and ('sar' not in configs or not configs['sar'])):
                    normalization = transforms.Normalize(mean=configs["mean"], std=configs["std"])
                #normalization = transforms.Normalize(mean=configs["mean"], std=configs["std"])
                image = normalization(image)
            
            if configs['dataset']=='bigearthnet' and configs['num_classes']==5:
                inverted_nomenclature_five = {2: 0, 14: 0, 0: 1, 16: 1, 3: 1, 6: 1, 15: 1, 13: 1, 17: 2, 10: 2, 12: 2, 1: 2, 5: 2, 7: 2, 18: 2, 8: 3, 9: 3, 4: 4, 11: 4}
                l5 = torch.zeros(5)
                for i,_ in enumerate(label):
                    l5[inverted_nomenclature_five[i]] = l5[inverted_nomenclature_five[i]] or label[i]
                label = l5
                
            if not yield_dataset:
                yield (image, label)
            else:
                yield (image, label, configs["dataset"])

    if "webdataset_root_path" not in configs or configs["webdataset_root_path"] is None:
        configs["webdataset_path"] = os.path.join(configs["root_path"], "webdataset", configs["dataset"])
    else:
        configs["webdataset_path"] = os.path.join(
            os.path.expandvars(configs["webdataset_root_path"]),
            "webdataset",
            configs["dataset"],
        )

    if not os.path.isdir(os.path.join(configs["webdataset_path"], "train")) or not os.path.isdir(
        os.path.join(configs["webdataset_path"], "val")
    ):  # or not os.path.isdir(os.path.join(configs['webdataset_path'],'test')):
        if not configs["webdataset_parallel"]:
            web_writer.wds_write(configs)
        else:
            web_writer.wds_write_parallel(configs)
        print("Created webdataset for: ", configs["dataset"])
        exit(1)

    max_train_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "train", "*.tar")))[-1]
    # Handle both naming patterns: sample-train-{index}.tar and sample-train-{process}-{index}.tar
    shard_name = os.path.basename(max_train_shard)
    parts = shard_name.replace(".tar", "").split("-train-")[-1].split("-")
    if len(parts) == 2:
        # Pattern: sample-train-{process}-{index}.tar (parallel writer)
        max_train_index = parts[1]  # Get the shard index
    else:
        # Pattern: sample-train-{index}.tar (sequential writer)
        max_train_index = parts[0]

    """if "dataset_percentage" in configs and configs["dataset_percentage"] is not None:
        max_train_index_int = int(int(max_train_index) * configs["dataset_percentage"])
        max_train_index = "{:06d}".format(max_train_index_int)"""

    # Get all train shard files (handles both sequential and parallel writer patterns)
    all_train_shards = sorted(glob.glob(os.path.join(configs["webdataset_path"], "train", "sample-train-*.tar")))
    if "dataset_percentage" in configs and configs["dataset_percentage"] is not None:
        # Shuffle samples with seed 222
        random.Random(222).shuffle(all_train_shards)
        train_shards = all_train_shards[: int(len(all_train_shards) * configs["dataset_percentage"])]
    else:
        train_shards = all_train_shards

    seed = configs.get("seed")
    _shuffle_rng = random.Random(seed) if seed is not None else None  # wds.shuffle uses Random(pid+time) when rng=None
    # Deterministic shard order: pre-shuffle with seed (WebDataset seed param not in all versions)
    if not self_supervised and not supervised_foundation:
        if seed is not None:
            shards_to_use = list(train_shards)
            random.Random(seed).shuffle(shards_to_use)
            train_dataset = wds.WebDataset(
                shards_to_use, shardshuffle=False, resampled=False
            ).shuffle(configs["webdataset_shuffle_size"], rng=_shuffle_rng)
        else:
            _ds = wds.WebDataset(train_shards, shardshuffle=True, resampled=False)
            train_dataset = _ds.shuffle(configs["webdataset_shuffle_size"], rng=_shuffle_rng) if _shuffle_rng else _ds.shuffle(configs["webdataset_shuffle_size"])
        train_dataset = train_dataset.compose(get_patches)
        train_dataset = train_dataset.batched(configs["batch_size"], partial=False)
    elif supervised_foundation:
        if seed is not None:
            shards_to_use = list(train_shards)
            random.Random(seed).shuffle(shards_to_use)
            train_dataset = wds.WebDataset(
                shards_to_use, shardshuffle=False, resampled=resample_shards
            ).shuffle(configs["webdataset_shuffle_size"], rng=_shuffle_rng)
        else:
            _ds = wds.WebDataset(train_shards, shardshuffle=True, resampled=resample_shards)
            train_dataset = _ds.shuffle(configs["webdataset_shuffle_size"], rng=_shuffle_rng) if _shuffle_rng else _ds.shuffle(configs["webdataset_shuffle_size"])
        train_dataset = train_dataset.compose(get_patches)
        train_dataset = train_dataset.batched(configs["batch_size"], partial=False)
    else:
        # Combine train, val and test shards for ssl training
        max_val_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "val", "*.tar")))[-1]
        max_val_index = max_val_shard.split("-val-")[-1][:-4]
        val_shards = os.path.join(
            configs["webdataset_path"],
            "val",
            "sample-val-{000000.." + max_val_index + "}.tar",
        )

        max_test_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "test", "*.tar")))[-1]
        max_test_index = max_test_shard.split("-test-")[-1][:-4]
        test_shards = os.path.join(
            configs["webdataset_path"],
            "test",
            "sample-test-{000000.." + max_test_index + "}.tar",
        )

        full_shards = list(braceexpand(train_shards)) + list(braceexpand(val_shards)) + list(braceexpand(test_shards))
        if resample_shards:
            train_dataset = wds.DataPipeline(
                wds.ResampledShards(full_shards),
                wds.tarfile_to_samples(),
                wds.shuffle(configs["webdataset_shuffle_size"], rng=_shuffle_rng) if _shuffle_rng else wds.shuffle(configs["webdataset_shuffle_size"]),
                get_patches,
                wds.batched(configs["batch_size"], partial=False),
            )
        else:
            if seed is not None:
                shards_to_use = list(full_shards)
                random.Random(seed).shuffle(shards_to_use)
                train_dataset = wds.WebDataset(
                    shards_to_use, shardshuffle=False, resampled=False
                ).shuffle(configs["webdataset_shuffle_size"], rng=_shuffle_rng)
            else:
                _ds = wds.WebDataset(full_shards, shardshuffle=True, resampled=False)
                train_dataset = _ds.shuffle(configs["webdataset_shuffle_size"], rng=_shuffle_rng) if _shuffle_rng else _ds.shuffle(configs["webdataset_shuffle_size"])
            train_dataset = train_dataset.compose(get_patches)
            train_dataset = train_dataset.batched(configs["batch_size"], partial=False)

    # Disable pin_memory for training WebDataset loaders because batches always include keys
    # Keys are strings (sample["__key__"]) and cannot be pinned to GPU memory
    # This causes CUDA errors: "CUDA error: invalid argument" in pin_memory thread
    # Eval/val loaders use get_patches_eval which don't return keys, so pin_memory works there
    use_pin_memory = False  # Always disable for training loaders due to keys in batch

    def worker_init_fn(worker_id):
        w_seed = (seed if seed is not None else 0) + worker_id
        np.random.seed(w_seed)
        random.seed(w_seed)
        torch.manual_seed(w_seed)

    train_loader = wds.WebLoader(
        train_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=use_pin_memory,
        prefetch_factor=configs["prefetch_factor"],
        persistent_workers=configs["persistent_workers"],
        worker_init_fn=worker_init_fn if configs["num_workers"] > 0 else None,
    )
    # REMOVED: Blocking double shuffle that causes GPU utilization spikes
    # The WebDataset already shuffles at line 677, so this second shuffle is redundant
    # and creates blocking synchronization points that cause GPU to wait
    # 
    # Original blocking code (commented out):
    # train_loader = (
    #     train_loader.unbatched()
    #     .shuffle(
    #         configs["webdataset_shuffle_size"],
    #         initial=configs["webdataset_initial_buffer"],
    #     )
    #     .batched(configs["batch_size"])
    # )
    if repeat:
        train_loader = train_loader.repeat()

    val_shards = sorted(glob.glob(os.path.join(configs["webdataset_path"], "val", "sample-val-*.tar")))

    val_dataset = wds.WebDataset(val_shards, shardshuffle=False, resampled=False)
    val_dataset = val_dataset.compose(get_patches_eval)
    val_dataset = val_dataset.batched(configs["batch_size"], partial=True)

    # Eval batches don't include keys (only image, label), so pin_memory works fine
    val_loader = wds.WebLoader(
        val_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=configs["pin_memory"],  # Safe for eval since no strings in batches
    )
    test_shards = sorted(glob.glob(os.path.join(configs["webdataset_path"], "test", "sample-test-*.tar")))

    test_dataset = wds.WebDataset(test_shards, shardshuffle=False, resampled=False)
    test_dataset = test_dataset.compose(get_patches_eval)
    test_dataset = test_dataset.batched(configs["batch_size"], partial=True)

    # Eval batches don't include keys (only image, label), so pin_memory works fine
    test_loader = wds.WebLoader(
        test_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=configs["pin_memory"],  # Safe for eval since no strings in batches
    )

    return train_loader, val_loader, test_loader


def create_universal_webdataset_loader(configs):
    loaders = []
    datasets = []
    for idx, dataset in enumerate(configs["train_datasets"]):
        print("Preparing loader for: ", dataset)
        current_config = configs.copy()
        current_config["dataset"] = dataset
        dataset_config_path = "configs/datasets/" + str(dataset).lower() + ".json"
        dataset_config = load_config_file(dataset_config_path)
        current_config.update(dataset_config)
        # Load normalization configs
        if current_config["normalization"] == "standard":
            stats_config_path = "configs/stats/stats.json"
            stats_config = load_config_file(stats_config_path)
            current_config["mean"] = stats_config[current_config["dataset"]]["mean"]
            current_config["std"] = stats_config[current_config["dataset"]]["std"]
            # Need to decide whether to share workers among datasets or split them equally
            if configs["split_workers"]:
                current_config["num_workers"] = configs["num_workers"] // len(configs["train_datasets"])
            
                print(
                    "Setting up loader for dataset: ",
                    dataset,
                    " with number of workers: ",
                    current_config["num_workers"],
                )
        train, _, _ = create_webdataset_loaders(current_config, repeat=False, self_supervised=True, resample_shards=True)
        loaders.append(train)
        datasets.append(dataset)
    return loaders, None, None


def create_universal_supervised_webdataset_loader(configs):
    loaders = []
    datasets = []
    for idx, dataset in enumerate(configs["train_datasets"]):
        print("Preparing loader for: ", dataset)
        current_config = configs.copy()
        current_config["dataset"] = dataset
        dataset_config_path = "configs/datasets/" + str(dataset).lower() + ".json"
        dataset_config = load_config_file(dataset_config_path)
        current_config.update(dataset_config)
        # Load normalization configs
        if current_config["normalization"] == "standard":
            stats_config_path = "configs/stats/stats.json"
            stats_config = load_config_file(stats_config_path)
            current_config["mean"] = stats_config[current_config["dataset"]]["mean"]
            current_config["std"] = stats_config[current_config["dataset"]]["std"]
            # Need to decide whether to share workers among datasets or split them equally
            if configs["split_workers"]:
                current_config["num_workers"] = configs["num_workers"] // len(configs["train_datasets"])
                print(
                    "Setting up loader for dataset: ",
                    dataset,
                    " with number of workers: ",
                    current_config["num_workers"],
                )
        print("Setting up loader for dataset: ", dataset)
        train, _, _ = create_webdataset_loaders(
            current_config,
            repeat=True,
            self_supervised=False,
            resample_shards=True,
            yield_dataset=True,
            supervised_foundation=True,
        )
        loaders.append(train)
        datasets.append(dataset)
    return loaders, None, None


def initialize_metrics(configs):
    metrics = []
    ign = configs.get("ignore_index", None)
    for metric in configs["metrics"]:
        for metric_aggregation_strategy in configs["metric_aggregation_strategy"]:
            if configs["multilabel"]:
                if metric.lower() == "accuracy":
                    m = Accuracy(
                        task="multilabel",
                        average=metric_aggregation_strategy,
                        multidim_average="global",
                        num_labels=configs["num_classes"],
                    ).to(configs["device"])
                elif metric.lower() == "fscore":
                    m = F1Score(
                        task="multilabel",
                        num_labels=configs["num_classes"],
                        average=metric_aggregation_strategy,
                        multidim_average="global",
                    ).to(configs["device"])
                elif metric.lower() == "precision":
                    m = Precision(
                        task="multilabel",
                        average=metric_aggregation_strategy,
                        num_labels=configs["num_classes"],
                        multidim_average="global",
                    ).to(configs["device"])
                elif metric.lower() == "recall":
                    m = Recall(
                        task="multilabel",
                        average=metric_aggregation_strategy,
                        num_labels=configs["num_classes"],
                        multidim_average="global",
                    ).to(configs["device"])
                elif metric.lower() == "iou":
                    m = JaccardIndex(
                        task="multilabel",
                        num_labels=configs["num_classes"],
                        average=metric_aggregation_strategy,
                    ).to(configs["device"])
                elif metric.lower() == "coverage":
                    m = MultilabelCoverageError(num_labels=configs["num_classes"]).to(configs["device"])
                elif metric.lower() == "map":
                    m = SafeAveragePrecision(
                        task="multilabel", 
                        num_labels=configs["num_classes"], 
                        average=metric_aggregation_strategy,
                        validate_args=False,
                        thresholds=None
                    ).to(configs["device"])
                else:
                    print("Metric: ", metric, " not supported!")
                    continue
            else:
                if metric.lower() == "accuracy":
                    m = Accuracy(
                        task="multiclass",
                        average=metric_aggregation_strategy,
                        multidim_average="global",
                        num_classes=configs["num_classes"],
                        ignore_index=ign,
                    ).to(configs["device"])
                elif metric.lower() == "fscore":
                    m = F1Score(
                        task="multiclass",
                        num_classes=configs["num_classes"],
                        average=metric_aggregation_strategy,
                        multidim_average="global",
                        ignore_index=ign,
                    ).to(configs["device"])
                elif metric.lower() == "precision":
                    m = Precision(
                        task="multiclass",
                        average=metric_aggregation_strategy,
                        num_classes=configs["num_classes"],
                        multidim_average="global",
                        ignore_index=ign,
                    ).to(configs["device"])
                elif metric.lower() == "recall":
                    m = Recall(
                        task="multiclass",
                        average=metric_aggregation_strategy,
                        num_classes=configs["num_classes"],
                        multidim_average="global",
                        ignore_index=ign,
                    ).to(configs["device"])
                elif metric.lower() == "iou":
                    m = JaccardIndex(
                        task="multiclass",
                        num_classes=configs["num_classes"],
                        average=metric_aggregation_strategy,
                        ignore_index=ign,
                    ).to(configs["device"])
                else:
                    print("Metric: ", metric, " not supported!")
                    continue
            metrics.append(m)
    # FLAIR paper metric: mIoU on first 12 classes only (indices 0–11), excluding "other" (12–18)
    # Class order per paper: building, pervious surface, impervious surface, bare soil, water,
    # coniferous, deciduous, brushwood, vineyard, herbaceous vegetation, agricultural land, plowed land.
    if configs.get("task") == "segmentation" and configs.get("dataset") == "flair" and configs.get("num_classes") == 19 and configs.get("flair_miou_12", True):
        m_12 = JaccardIndex(task="multiclass", num_classes=19, average="none").to(configs["device"])
        m_12.flair_12 = True  # mark for logging mean of first 12 classes only
        metrics.append(m_12)
    # When ignore_index is set (e.g. MARIDA unlabeled=11): add macro IoU over all classes except that one (paper-style metric).
    # Omit ignore_index in config for datasets like FLAIR to use standard loss/metrics without excluding any class.
    if configs.get("task") == "segmentation" and configs.get("ignore_index") is not None:
        ign = configs["ignore_index"]
        m_ignore = JaccardIndex(
            task="multiclass", num_classes=configs["num_classes"], average="none", ignore_index=ign
        ).to(configs["device"])
        m_ignore.exclude_ignore_index = ign  # for logging: mean over classes != ign
        metrics.append(m_ignore)
    return tuple(metrics)


def compute_class_weights_log_inverse(
    loader,
    num_classes,
    device,
    epsilon=1.02,
    max_weight=10.0,
    max_batches=None,
    ignore_index=None,
):
    """
    Compute class weights from training set pixel frequencies using log-inverse weighting.
    weight[c] = 1 / log(freq[c] + epsilon). Compresses extreme ratios and boosts rare
    classes safely. Recommended for macro Jaccard / IoU with severe class imbalance.

    Args:
        loader: DataLoader yielding (image, label, ...); label shape [B, H, W].
        num_classes: Number of segmentation classes (0..num_classes-1).
        device: Device for the returned tensor.
        epsilon: Add to freq before log; 1.02--1.1 keeps weights stable (recommended 1.02).
        max_weight: Cap weights so max(weight) <= max_weight (sanity: <= 10, ideally 2--6).
        max_batches: If set, use at most this many batches for counting (faster; None = full pass).

    Returns:
        weights: Tensor of shape (num_classes,) on device, dtype float32.
    """
    import math
    count = [0] * num_classes
    total = 0  # number of valid (non-ignore) pixels
    n_batches = 0
    for batch in loader:
        label = batch[1]
        if not isinstance(label, torch.Tensor):
            continue
        label = label.long()
        # Valid pixels are those within [0, num_classes) and, optionally, not equal to ignore_index
        valid = (label >= 0) & (label < num_classes)
        if ignore_index is not None:
            valid = valid & (label != ignore_index)

        for c in range(num_classes):
            if ignore_index is not None and c == ignore_index:
                # Skip ignore_index class entirely when accumulating counts
                continue
            count[c] += ((label == c) & valid).sum().item()
        total += valid.sum().item()
        n_batches += 1
        if n_batches % 50 == 0:
            print("[class_weights] scanning batches: {} ({} pixels so far)".format(n_batches, total), flush=True)
        if max_batches is not None and n_batches >= max_batches:
            break
    total = max(total, 1)
    freq = [count[c] / total for c in range(num_classes)]
    # Progress: report once after counting
    print("[class_weights] counted {} valid pixels in {} batches.".format(total, n_batches))
    # Log-inverse: weight[c] = 1 / log(freq[c] + epsilon)
    weights = []
    for c in range(num_classes):
        f = max(freq[c], 1e-8)
        w = 1.0 / math.log(f + epsilon)
        weights.append(w)
    w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    # Cap so max(weight) <= max_weight
    if w_tensor.max() > max_weight:
        w_tensor = w_tensor.clamp(max=max_weight)
    # Optional: scale so max is in 2--6 range for stability (user can disable via max_weight=10)
    return w_tensor


# MARIDA paper-style static class weights (log-inverse on pixel distribution), after
# aggregating the 4 water-related classes into Marine Water as done in their U-Net code
# (see Kakogeorgiou et al., train.py in the MARIDA repository).
# These 11 values correspond to classes 0..10 (thematic); class 11 is unlabeled/ignore.
MARIDA_CLASS_DISTR_AGG = torch.tensor(
    [0.00452, 0.00203, 0.00254, 0.00168, 0.00766,
     0.15206, 0.23457, 0.35941, 0.00109, 0.20218, 0.03226],
    dtype=torch.float32,
)


def marida_paper_class_weights(device, weight_param=1.03):
    """
    Return MARIDA paper-style class weights for segmentation (11 thematic classes).
    weight[c] = 1 / log(weight_param + class_distr[c]), where class_distr is the
    aggregated pixel distribution after merging water subclasses, matching the
    original MARIDA U-Net training code.

    The returned tensor has shape (12,), where indices 0..10 correspond to thematic
    classes and index 11 (ignore/unlabeled) has neutral weight 1.0.
    """
    distr = MARIDA_CLASS_DISTR_AGG.to(device=device)
    # 11 weights for classes 0..10
    w11 = 1.0 / torch.log(weight_param + distr)
    # Extend to 12 (add neutral weight for ignore class 11)
    w12 = torch.ones(12, dtype=torch.float32, device=device)
    w12[:11] = w11
    return w12


class MulticlassDiceLoss(nn.Module):
    """Multiclass Dice loss (soft Dice, macro over classes). Aligns with Jaccard/IoU.
    ignore_index: if set, that class is excluded from loss (e.g. unlabeled); only other classes are averaged."""

    def __init__(self, num_classes, smooth=1e-6, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: [B, C, H, W], targets: [B, H, W] class indices
        probs = torch.softmax(logits, dim=1)
        valid = (targets >= 0) & (targets < self.num_classes)
        if self.ignore_index is not None:
            valid = valid & (targets != self.ignore_index)
        valid = valid.float()
        dice_per_class = []
        classes_to_use = [c for c in range(self.num_classes) if c != self.ignore_index]
        if not classes_to_use:
            return logits.new_zeros(1).squeeze()
        for c in classes_to_use:
            pred_c = probs[:, c] * valid
            target_c = ((targets == c).float() * valid)
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_c = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice_c)
        dice_mean = torch.stack(dice_per_class).mean()
        return 1.0 - dice_mean


class CombinedCEPlusDiceLoss(nn.Module):
    """Cross-Entropy + Dice. Optional mask for CE (e.g. redundancy masking).
    ignore_index: if set, those target pixels are excluded from both CE and Dice (e.g. MARIDA unlabeled=11)."""

    def __init__(self, num_classes, dice_weight=0.5, dice_smooth=1e-6, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.dice = MulticlassDiceLoss(num_classes, smooth=dice_smooth, ignore_index=ignore_index)

    def forward(self, logits, targets, mask=None):
        # logits: [B, C, H, W], targets: [B, H, W]
        ce_kw = {"ignore_index": self.ignore_index} if self.ignore_index is not None else {}
        if mask is not None:
            ce_per_pixel = torch.nn.functional.cross_entropy(
                logits, targets.long(), reduction="none", **ce_kw
            )
            mask_float = mask.float()
            if mask_float.dim() == 2:
                mask_float = mask_float.unsqueeze(0).expand_as(ce_per_pixel)
            elif mask_float.dim() == 3 and mask_float.shape != ce_per_pixel.shape:
                mask_float = torch.nn.functional.interpolate(
                    mask_float.unsqueeze(1).float(),
                    size=ce_per_pixel.shape[-2:],
                    mode="nearest",
                ).squeeze(1)
            if self.ignore_index is not None:
                mask_float = mask_float * (targets != self.ignore_index).float()
            ce = (ce_per_pixel * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        else:
            ce = torch.nn.functional.cross_entropy(logits, targets.long(), **ce_kw)
        dice = self.dice(logits, targets)
        return ce + self.dice_weight * dice


class WeightedCEDiceLoss(nn.Module):
    """
    Class-weighted Cross-Entropy + Dice for macro Jaccard / IoU.
    Weights computed e.g. via log-inverse: weight[c] = 1 / log(freq[c] + epsilon).
    Same interface as CombinedCEPlusDiceLoss (supports mask= for redundancy masking).
    ignore_index: if set, those target pixels are excluded (e.g. MARIDA unlabeled=11).
    Loss = (1 - dice_weight) * WeightedCE + dice_weight * Dice.
    """

    def __init__(self, num_classes, weights, dice_weight=0.5, dice_smooth=1e-6, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", weights.float())
        self.dice = MulticlassDiceLoss(num_classes, smooth=dice_smooth, ignore_index=ignore_index)

    def forward(self, logits, targets, mask=None):
        # logits: [B, C, H, W], targets: [B, H, W]; weights on same device as logits
        w = self.class_weights.to(logits.device)
        ce_kw = {"weight": w}
        if self.ignore_index is not None:
            ce_kw["ignore_index"] = self.ignore_index
        if mask is not None:
            ce_per_pixel = torch.nn.functional.cross_entropy(
                logits, targets.long(), reduction="none", **ce_kw
            )
            mask_float = mask.float()
            if mask_float.dim() == 2:
                mask_float = mask_float.unsqueeze(0).expand_as(ce_per_pixel)
            elif mask_float.dim() == 3 and mask_float.shape != ce_per_pixel.shape:
                mask_float = torch.nn.functional.interpolate(
                    mask_float.unsqueeze(1).float(),
                    size=ce_per_pixel.shape[-2:],
                    mode="nearest",
                ).squeeze(1)
            if self.ignore_index is not None:
                mask_float = mask_float * (targets != self.ignore_index).float()
            ce = (ce_per_pixel * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        else:
            ce = torch.nn.functional.cross_entropy(logits, targets.long(), **ce_kw)
        dice = self.dice(logits, targets)
        return (1.0 - self.dice_weight) * ce + self.dice_weight * dice


class MaridaWeightedBCELoss(nn.Module):
    """
    MARIDA-style per-class weighted BCE for single-label segmentation.
    - logits:  [B, C, H, W]
    - targets: [B, H, W] with values in {0..num_classes-1}; ignore_index optionally excluded.
    Internally builds per-class one-hot masks and applies BCEWithLogitsLoss with per-class pos_weight.
    """

    def __init__(self, num_classes, pos_weights, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # Register as buffer so it moves with the model and is saved in state_dict
        self.register_buffer("pos_weights", pos_weights.float())

    def forward(self, logits, targets):
        # logits: [B, C, H, W], targets: [B, H, W]
        B, C, H, W = logits.shape
        assert C == self.num_classes, f"Expected C={self.num_classes}, got {C}"

        # Build valid mask (exclude ignore_index if provided)
        valid = (targets >= 0) & (targets < self.num_classes)
        if self.ignore_index is not None:
            valid = valid & (targets != self.ignore_index)
        if not valid.any():
            # No valid pixels in this batch; return zero to avoid NaNs
            return logits.new_zeros([])

        # Flatten spatial dims so we don't care about H/W ordering
        # logits_flat: [N, C], targets_flat: [N], valid_flat: [N]
        logits_flat = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets_flat = targets.contiguous().view(-1)
        valid_flat = valid.contiguous().view(-1)

        logits_flat = logits_flat[valid_flat]
        targets_flat = targets_flat[valid_flat]

        if logits_flat.numel() == 0:
            return logits.new_zeros([])

        # One-hot targets [N, C]
        targets_oh = torch.nn.functional.one_hot(
            targets_flat.clamp(0, self.num_classes - 1),
            num_classes=self.num_classes,
        ).to(dtype=logits.dtype, device=logits.device)

        pos_w = self.pos_weights.to(logits.device)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_flat, targets_oh, pos_weight=pos_w
        )
        return bce


def create_loss(configs):
    # Optional ignore_index: exclude a target class from loss (e.g. MARIDA unlabeled=11). Set in dataset/training config; omit for FLAIR etc.
    ignore_index = configs.get("ignore_index")
    if configs["loss"] == "cross_entropy":
        if configs["multilabel"]:
            return nn.BCEWithLogitsLoss()
        else:
            kwargs = {} if ignore_index is None else {"ignore_index": ignore_index}
            return nn.CrossEntropyLoss(**kwargs)
    elif configs["loss"] == "ce_dice":
        num_classes = configs.get("num_classes")
        if num_classes is None:
            raise ValueError("num_classes required in config for ce_dice loss")
        dice_weight = configs.get("dice_weight", 0.5)
        return CombinedCEPlusDiceLoss(
            num_classes=num_classes,
            dice_weight=float(dice_weight),
            ignore_index=ignore_index,
        )
    elif configs["loss"] == "ce_dice_weighted":
        num_classes = configs.get("num_classes")
        if num_classes is None:
            raise ValueError("num_classes required in config for ce_dice_weighted loss")
        dice_weight = configs.get("dice_weight", 0.5)
        class_weights = configs.get("class_weights")
        if class_weights is None:
            # e.g. test-only run without precomputed weights: fall back to unweighted CE+Dice
            print("[ce_dice_weighted] class_weights not in config; using unweighted CE+Dice (e.g. test-only).")
            return CombinedCEPlusDiceLoss(
                num_classes=num_classes,
                dice_weight=float(dice_weight),
                ignore_index=ignore_index,
            )
        return WeightedCEDiceLoss(
            num_classes=num_classes,
            weights=class_weights,
            dice_weight=float(dice_weight),
            ignore_index=ignore_index,
        )
    elif configs["loss"] == "marida_bce":
        # MARIDA-only BCE baseline with per-class pos_weights stored in configs["marida_pos_weights"]
        if configs.get("dataset") != "marida":
            raise ValueError("marida_bce loss is only supported for dataset='marida'")
        num_classes = configs.get("num_classes")
        if num_classes is None:
            raise ValueError("num_classes required in config for marida_bce loss")
        pos_weights = configs.get("marida_pos_weights")
        if pos_weights is None:
            raise ValueError("marida_pos_weights not set; compute it in training.segmentation.train() before create_loss")
        return MaridaWeightedBCELoss(
            num_classes=num_classes,
            pos_weights=pos_weights,
            ignore_index=ignore_index,
        )
    else:
        print("Loss not supported")
        exit(3)


def create_optimizer(configs):
    if configs["optimizer"] == "adam":
        return torch.optim.AdamW
    elif configs["optimizer"] == "sgd":
        return torch.optim.SGD
    else:
        print("Optimizer not supported")


def create_procedures(configs):
    if configs["task"] == "classification":
        import training.classification as classification
        trainer = classification.train
        tester = classification.test
    elif configs["task"] == "segmentation":
        import training.segmentation as segmentation
        trainer = segmentation.train
        tester = segmentation.test
    else:
        print("AAAAAA")
        print("Task: ", configs["task"], " not supported.")
        exit(3)

    return trainer, tester


def count_params(model):
    """Count trainable parameters of a PyTorch Model"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    return nb_params


# ---------- Timing / gains metrics for wandb (redundancy investigation) ----------
def get_gpu_util(device):
    """Return current GPU utilization % (0-100) or None. Used for train/gpu_util_pct in wandb."""
    if not torch.cuda.is_available():
        return None
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type != "cuda":
        return None
    idx = dev.index if dev.index is not None else 0
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        return pynvml.nvmlDeviceGetUtilizationRates(h).gpu
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "-i", str(idx), "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().split("\n")[0].strip())
    except Exception:
        pass
    return None


# ViT backbone -> (embed_dim, num_heads, num_layers, patch_size) for GFLOPs estimate
VIT_BACKBONE_PARAMS = {
    "vit_tiny_patch16_224": (192, 3, 12, 16),
    "vit_small_patch16_224": (384, 6, 12, 16),
    "vit_small_patch32_224": (384, 6, 12, 32),
    "vit_base_patch16_224": (768, 12, 12, 16),
    "vit_base_patch32_224": (768, 12, 12, 32),
    "vit_base_patch8_224": (768, 12, 12, 8),
    "vit_large_patch16_224": (1024, 16, 24, 16),
    "vit_base_patch14_dinov2.lvd142m": (768, 12, 12, 14),
    "vit_small_patch14_dinov2": (384, 6, 12, 14),
    "vit_large_patch14_dinov2": (1024, 16, 24, 14),
}


def estimate_vit_encoder_gflops(image_size, in_channels, retain_percent, backbone_name, batch_size=1):
    """
    Estimate encoder GFLOPs (billions of FLOPs) for one forward pass with given retain_percent.
    Uses standard ViT formulas: patch embed + num_layers * (QKV + attention + proj + MLP).
    N_eff = num_patches * retain_percent/100; FLOPs scale with N_eff (and N_eff^2 for attention).
    Convention: 1 MAC = 1 FLOP (matches fvcore FlopCountAnalysis, so encoder_gflops aligns with measured full_gflops).
    Returns GFLOPs per sample (batch_size=1) or total for batch; None if backbone unknown.
    """
    bn = backbone_name.split(".")[0].strip().lower() if isinstance(backbone_name, str) else ""
    if bn not in VIT_BACKBONE_PARAMS:
        return None
    embed_dim, num_heads, num_layers, patch_size = VIT_BACKBONE_PARAMS[bn]
    # Use integer division so non-divisible image_size (e.g. BigEarthNet 120 with patch_size 16) still get an estimate
    num_patches = (image_size // patch_size) ** 2
    if num_patches < 1:
        return None
    n_eff = max(1, int(num_patches * retain_percent / 100.0))
    # Patch embed: 1 MAC per output element
    flops_patch = in_channels * (patch_size ** 2) * num_patches * embed_dim
    # Per block: QKV 3*N*embed_dim^2, attn 2*N^2*embed_dim (2 matmuls), proj N*embed_dim^2, MLP 8*N*embed_dim^2
    flops_per_block = (
        3 * n_eff * (embed_dim ** 2)
        + 2 * (n_eff ** 2) * embed_dim
        + n_eff * (embed_dim ** 2)
        + 8 * n_eff * (embed_dim ** 2)
    )
    flops_total = (flops_patch + num_layers * flops_per_block) * batch_size
    return flops_total / 1e9  # GFLOPs


def estimate_full_architecture_gflops(image_size, in_channels, retain_percent, backbone_name, batch_size=1,
                                       architecture="rvitupernet", pyramid_channels=256):
    """
    Estimate full-architecture GFLOPs (encoder + decoder for segmentation), same style as encoder_gflops.
    For segmentation (rvitupernet): encoder_gflops + rough UPerNet decoder estimate.
    For classification: encoder only, so returns encoder_gflops.
    Log the result as full_gflops in wandb alongside encoder_gflops.
    """
    encoder_gflops = estimate_vit_encoder_gflops(
        image_size, in_channels, retain_percent, backbone_name, batch_size=batch_size
    )
    if encoder_gflops is None:
        return None
    if architecture != "rvitupernet":
        return encoder_gflops  # classification: full model is encoder only
    # UPerNet decoder: rough FLOP estimate (projs + PSP + FPN + bottleneck + head)
    bn = backbone_name.split(".")[0].strip().lower() if isinstance(backbone_name, str) else ""
    if bn not in VIT_BACKBONE_PARAMS:
        return encoder_gflops
    _, _, _, patch_size = VIT_BACKBONE_PARAMS[bn]
    num_patches = (image_size // patch_size) ** 2
    if num_patches < 1:
        return encoder_gflops
    # Decoder: projs (1x1) + PSP/FPN/bottleneck/head. Channel list ~ [embed_dim, embed_dim/2, embed_dim/4, embed_dim/8]
    embed_dim = VIT_BACKBONE_PARAMS[bn][0]
    ch = [embed_dim, max(64, embed_dim // 2), max(64, embed_dim // 4), max(64, embed_dim // 8)]
    # Proj convs: 1 MAC per output (1 MAC = 1 FLOP to match fvcore)
    flops_proj = batch_size * num_patches * embed_dim * sum(ch)
    # PSP + FPN + bottleneck + head: rough spatial * pyramid^2 * ops
    height = image_size // patch_size
    spatial = max(1, height) ** 2
    flops_dec = batch_size * spatial * (pyramid_channels ** 2) * 25  # ~25 conv-like ops
    flops_decoder = (flops_proj + flops_dec) / 1e9
    return encoder_gflops + flops_decoder


def make_gflop_measurement_inputs(configs, device, task="segmentation"):
    """
    Create synthetic inputs with fixed shape from configs for consistent GFLOP measurement.
    Use this so train and test measure the same forward path (avoid different batch shapes).
    """
    batch_size = configs.get("batch_size", 1)
    in_channels = configs.get("in_channels", 3)
    img_size = configs.get("effective_image_size", configs.get("image_size", 224))
    image = torch.zeros(batch_size, in_channels, img_size, img_size, device=device, dtype=torch.float32)
    if task == "segmentation":
        label = torch.zeros(batch_size, img_size, img_size, dtype=torch.long, device=device)
        keys = [f"gflop_{i}" for i in range(batch_size)]
        return image, {"label": label, "keys": keys}
    return image, {}


def estimate_full_model_gflops(model, *input_tensors, input_kwargs=None, measure_training_path=False):
    """
    Measure full-model GFLOPs via fvcore (or thop fallback).
    Returns (full_gflops, encoder_gflops_fv) where encoder_gflops_fv is fvcore-measured encoder
    GFLOPs (None if unavailable).

    measure_training_path: If True, use model.train() so decoder takes scatter path (training cost).
    If False, use model.eval() so decoder takes inference path (scatter when masking+retain<100%,
    direct when retain=100% or no masking).
    """
    if not input_tensors:
        return None, None

    input_kwargs = input_kwargs or {}
    original_mode = model.training
    if measure_training_path:
        model.train()
    else:
        model.eval()

    # Force same code path for train and test: set testing=True and apply_mask_in_infer=True
    # when model has masking (encoder or retain_percent). Training had use_mask=False -> different
    # path (no padding, no cos_sim). Test had use_mask when apply_infer -> different FLOPs.
    root_model = getattr(model, "module", model)
    _saved = {}
    for m, name in [(root_model, "root"), (getattr(root_model, "encoder", None), "encoder"),
                    (getattr(root_model, "model", None), "model")]:
        if m is None:
            continue
        has_masking = (getattr(m, "retain_percent", 100) <= 100 or
                       hasattr(root_model, "encoder"))  # segmentation has encoder
        if has_masking:
            _saved[name] = (getattr(m, "testing", False), getattr(m, "apply_mask_in_infer", False))
            setattr(m, "testing", True)
            setattr(m, "apply_mask_in_infer", True)

    def _restore_masking_state():
        for m, name in [(root_model, "root"), (getattr(root_model, "encoder", None), "encoder"),
                        (getattr(root_model, "model", None), "model")]:
            if m is not None and name in _saved:
                setattr(m, "testing", _saved[name][0])
                setattr(m, "apply_mask_in_infer", _saved[name][1])

    try:
        from fvcore.nn import FlopCountAnalysis
        from fvcore.nn.jit_handles import elementwise_flop_counter, get_shape

        with torch.no_grad():
            if input_kwargs:
                class _Wrapper(torch.nn.Module):
                    def __init__(self, m, kwargs):
                        super().__init__()
                        self._m = m
                        self._kwargs = kwargs
                    def forward(self, *args):
                        return self._m(*args, **self._kwargs)
                target_module = _Wrapper(model, input_kwargs)
            else:
                target_module = model

            flops_counter = FlopCountAnalysis(target_module, input_tensors)

            def _sdpa_flop_jit(inputs, outputs):
                q_shape = get_shape(inputs[0])
                k_shape = get_shape(inputs[1])
                if not q_shape or not k_shape or len(q_shape) < 4:
                    return 0
                B, H, L, D = q_shape
                S = k_shape[2]
                return 4 * B * H * L * S * D + 5 * B * H * L * S

            def _linalg_vector_norm_flop_jit(inputs, outputs):
                """~2*dim FLOPs per output element (squares + sum + sqrt)."""
                inp_shape = get_shape(inputs[0])
                out_shape = get_shape(outputs[0])
                if not inp_shape or not out_shape:
                    return 0
                inp_numel = 1
                for s in inp_shape:
                    inp_numel *= s
                out_numel = 1
                for s in out_shape:
                    out_numel *= s
                dim_size = max(1, inp_numel // out_numel)
                return out_numel * (2 * dim_size + 1)  # squares + sum + sqrt

            def _zero_flop_jit(inputs, outputs):
                """Suppress warning; op contributes negligible FLOPs."""
                return 0

            handlers = {
                "aten::add": elementwise_flop_counter(0, 1),
                "aten::rsub": elementwise_flop_counter(0, 1),
                "aten::pad": elementwise_flop_counter(0, 0),
                "aten::gelu": elementwise_flop_counter(0, 4),
                "aten::scaled_dot_product_attention": _sdpa_flop_jit,
                # Extra handlers to reduce fvcore warnings (random_topx%, cosine sim, etc.)
                "aten::mul": elementwise_flop_counter(0, 1),
                "aten::mul_": elementwise_flop_counter(0, 1),
                "aten::div": elementwise_flop_counter(0, 1),
                "aten::abs": elementwise_flop_counter(0, 1),
                "aten::linalg_vector_norm": _linalg_vector_norm_flop_jit,
                "aten::sort": _zero_flop_jit,
                "aten::randperm": _zero_flop_jit,
            }
            flops_counter.set_op_handle(**handlers)
            flops_counter.uncalled_modules_warnings(False)

            full_gflops = flops_counter.total() / 1e9

            # Encoder-only GFLOPs
            encoder_gflops_fv = None

            # Method 1: Direct encoder trace (segmentation: model.encoder)
            root_model = getattr(model, "module", model)  # unwrap DDP/DP
            if hasattr(root_model, "encoder"):
                for use_kwargs in [True, False]:  # try with kwargs first, then without
                    try:
                        if use_kwargs and input_kwargs:
                            class _EncWrapper(torch.nn.Module):
                                def __init__(self, enc, kwargs):
                                    super().__init__()
                                    self._enc = enc
                                    self._kwargs = kwargs
                                def forward(self, *args):
                                    return self._enc(*args, **self._kwargs)
                            enc_mod = _EncWrapper(root_model.encoder, input_kwargs)
                        else:
                            enc_mod = root_model.encoder
                        enc_counter = FlopCountAnalysis(enc_mod, input_tensors)
                        enc_counter.set_op_handle(**handlers)
                        enc_counter.uncalled_modules_warnings(False)
                        encoder_gflops_fv = enc_counter.total() / 1e9
                        break
                    except Exception as e:
                        if os.environ.get("DEBUG_ENCODER_GFLOPS"):
                            import logging
                            logging.getLogger(__name__).warning(
                                "encoder_gflops_fv trace failed (use_kwargs=%s): %s", use_kwargs, e
                            )
                        continue

            # Method 2: total(module_name) on full-model trace
            if encoder_gflops_fv is None:
                encoder_prefixes = ("_m.model", "model", "_m.encoder", "encoder")
                for enc_path in encoder_prefixes:
                    try:
                        enc_total = flops_counter.total(enc_path)
                        if "model" in enc_path:
                            for head in (".fc", ".head"):
                                try:
                                    enc_total -= flops_counter.total(enc_path + head)
                                except KeyError:
                                    pass
                        encoder_gflops_fv = enc_total / 1e9
                        break
                    except KeyError:
                        continue

            # Method 3: Fallback from by_module()
            if encoder_gflops_fv is None:
                try:
                    by_mod = flops_counter.by_module()
                    best_enc, best_flops = None, 0
                    for mod_name, enc_total in by_mod.items():
                        if enc_total <= 0:
                            continue
                        if mod_name == "encoder" or mod_name.endswith(".encoder"):
                            if enc_total > best_flops:
                                best_enc, best_flops = mod_name, enc_total
                    if best_enc is not None:
                        encoder_gflops_fv = best_flops / 1e9
                except Exception:
                    pass

            _restore_masking_state()
            model.train(original_mode)
            return full_gflops, encoder_gflops_fv

    except Exception:
        _restore_masking_state()
        model.train(original_mode)
    try:
        import thop
        if measure_training_path:
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            if input_kwargs:
                class _Wrapper(torch.nn.Module):
                    def __init__(self, m, kwargs):
                        super().__init__()
                        self._m = m
                        self._kwargs = kwargs
                    def forward(self, *args):
                        return self._m(*args, **self._kwargs)
                wrapper = _Wrapper(model, input_kwargs)
                out = thop.profile(wrapper, inputs=input_tensors, verbose=False)
            else:
                out = thop.profile(model, inputs=input_tensors, verbose=False)
            if out is None:
                _restore_masking_state()
                return None, None
            flops, _ = out
        _restore_masking_state()
        model.train(original_mode)
        return flops / 1e9, None  # thop cannot provide encoder-only
    except Exception:
        _restore_masking_state()
        model.train(original_mode)
    return None, None


def get_peak_memory_mb(device):
    """Return peak GPU memory allocated (MB) for the given device. Uses the correct CUDA device so
    peak is not 0 when training on a non-default GPU (e.g. cuda:4). Returns None if not CUDA."""
    if not torch.cuda.is_available():
        return None
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type != "cuda":
        return None
    try:
        return torch.cuda.max_memory_allocated(device=dev) / (1024 ** 2)
    except Exception:
        return None


def reset_peak_memory_stats(device):
    """Reset CUDA peak memory stats for the given device so next get_peak_memory_mb is per-epoch."""
    if not torch.cuda.is_available():
        return
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(dev)
        except Exception:
            pass


def build_timing_log_dict(step_ms, full_forward_ms, peak_mem_mb, batch_size, retain_percent,
                          encoder_ms=None, encoder_peak_mem_mb=None, gpu_util_pct=None, encoder_gflops=None,
                          full_gflops=None, encoder_gflops_fv=None,
                          baseline_encoder_throughput=None, baseline_encoder_peak_mem=None):
    """Build wandb log dict for redundancy/gains metrics. Prefix keys with train/ for clarity."""
    d = {
        "train/step_ms": step_ms,
        "train/full_forward_ms": full_forward_ms,
        "train/peak_mem_mb": peak_mem_mb,
        "train/throughput_samples_per_sec": batch_size / (step_ms / 1000.0) if step_ms and step_ms > 0 else None,
        "train/retain_percent": retain_percent,
    }
    if encoder_ms is not None:
        d["train/encoder_ms"] = encoder_ms
        if batch_size and batch_size > 0 and encoder_ms > 0:
            enc_tp = batch_size * 1000.0 / encoder_ms
            d["train/encoder_throughput_samples_per_sec"] = enc_tp
            if baseline_encoder_throughput and baseline_encoder_throughput > 0:
                d["train/encoder_throughput_gain"] = enc_tp / baseline_encoder_throughput
    if encoder_peak_mem_mb is not None:
        d["train/encoder_peak_mem_mb"] = encoder_peak_mem_mb
        if baseline_encoder_peak_mem and baseline_encoder_peak_mem > 0 and encoder_peak_mem_mb > 0:
            d["train/encoder_peak_mem_gain"] = baseline_encoder_peak_mem / encoder_peak_mem_mb
    if gpu_util_pct is not None:
        d["train/gpu_util_pct"] = gpu_util_pct
    if encoder_gflops is not None:
        d["train/encoder_gflops"] = encoder_gflops
    if full_gflops is not None:
        d["train/full_gflops"] = full_gflops
        if batch_size and batch_size > 0:
            d["train/full_gflops_per_sample"] = full_gflops / batch_size
    if encoder_gflops_fv is not None:
        d["train/encoder_gflops_fv"] = encoder_gflops_fv
        if batch_size and batch_size > 0:
            d["train/encoder_gflops_fv_per_sample"] = encoder_gflops_fv / batch_size
    return {k: v for k, v in d.items() if v is not None}


def build_inference_timing_log_dict(inference_ms, peak_mem_mb, batch_size, retain_percent,
                                    encoder_ms=None, encoder_peak_mem_mb=None, gpu_util_pct=None,
                                    encoder_gflops=None, full_gflops=None, encoder_gflops_fv=None,
                                    baseline_encoder_throughput=None, baseline_encoder_peak_mem=None,
                                    prefix="test"):
    """Build wandb log dict for inference-phase metrics (when apply_mask_in_infer true). Prefix keys with test/ by default."""
    d = {
        f"{prefix}/inference_ms": inference_ms,
        f"{prefix}/throughput_samples_per_sec": batch_size / (inference_ms / 1000.0) if inference_ms and inference_ms > 0 else None,
        f"{prefix}/peak_mem_mb": peak_mem_mb,
        f"{prefix}/retain_percent": retain_percent,
    }
    if encoder_ms is not None:
        d[f"{prefix}/encoder_ms"] = encoder_ms
        if batch_size and batch_size > 0 and encoder_ms > 0:
            enc_tp = batch_size * 1000.0 / encoder_ms
            d[f"{prefix}/encoder_throughput_samples_per_sec"] = enc_tp
            if baseline_encoder_throughput and baseline_encoder_throughput > 0:
                d[f"{prefix}/encoder_throughput_gain"] = enc_tp / baseline_encoder_throughput
    if encoder_peak_mem_mb is not None:
        d[f"{prefix}/encoder_peak_mem_mb"] = encoder_peak_mem_mb
        if baseline_encoder_peak_mem and baseline_encoder_peak_mem > 0 and encoder_peak_mem_mb > 0:
            d[f"{prefix}/encoder_peak_mem_gain"] = baseline_encoder_peak_mem / encoder_peak_mem_mb
    if gpu_util_pct is not None:
        d[f"{prefix}/gpu_util_pct"] = gpu_util_pct
    if encoder_gflops is not None:
        d[f"{prefix}/encoder_gflops"] = encoder_gflops
    if full_gflops is not None:
        d[f"{prefix}/full_gflops"] = full_gflops
        if batch_size and batch_size > 0:
            d[f"{prefix}/full_gflops_per_sample"] = full_gflops / batch_size
    if encoder_gflops_fv is not None:
        d[f"{prefix}/encoder_gflops_fv"] = encoder_gflops_fv
        if batch_size and batch_size > 0:
            d[f"{prefix}/encoder_gflops_fv_per_sample"] = encoder_gflops_fv / batch_size
    return {k: v for k, v in d.items() if v is not None}


def compute_timing_summary(accumulator_list, include_std=True):
    """
    Compute mean (and optionally std) over a list of timing log dicts from training.
    Returns a dict with keys like train/step_ms_mean, train/step_ms_std, etc.,
    suitable for wandb.run.summary so runs can be compared in the Runs table.
    """
    if not accumulator_list:
        return {}
    summary = {}
    keys = list(accumulator_list[0].keys())
    for k in keys:
        vals = [d[k] for d in accumulator_list if k in d and d[k] is not None]
        if not vals:
            continue
        vals = np.array(vals, dtype=np.float64)
        summary[f"{k}_mean"] = float(np.nanmean(vals))
        if include_std and len(vals) > 1:
            summary[f"{k}_std"] = float(np.nanstd(vals))
    return summary