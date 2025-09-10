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
import torchvision
import torchvision.transforms as transforms
import webdataset as wds
from braceexpand import braceexpand
from torchmetrics import Accuracy, F1Score, JaccardIndex, Precision, Recall, AveragePrecision
from torchmetrics.classification import MultilabelCoverageError
import warnings
import albumentations as A

import datasets
import training.classification as classification
import training.segmentation as segmentation

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
    checkpoint_path = checkpoint_path / configs["wandb_run_name"]
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

    # Load normalization configs
    if config["normalization"] == "standard":
        stats_config_path = "configs/stats/stats.json"
        stats_config = load_config_file(stats_config_path)
        config["mean"] = stats_config[config["dataset"]]["mean"]
        config["std"] = stats_config[config["dataset"]]["std"]

    # Load augmentation configurations
    if config["augment"]:
        augmentation_configs_path = args.augmentation_config
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
            "modality_channels": multimodal_mae_config["modality_channels"],
            "dataset_modality_index": multimodal_mae_config["dataset_modality_index"],
        }
        config.update(mae_config)
        if "multiple_transformation" in config and config["multiple_transformation"]:
            config["backbone"] = "FoMo-Multi"
            config["architecture"] = "FoMoNet"
        else:
            config["backbone"] = "FoMo-Single"
            config["architecture"] = "FoMoNet"
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
    elif dataset_name.lower() == "bigearthnet":
        dataset = datasets.BigEarthNetDataset.BigEarthNetDataset(configs, mode)
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

    if configs["task"] == "mae":
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

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
                image = einops.rearrange(image, "h w c-> c h w")
                label[label==-1] = 11
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
                        b = image[2,:,:]
                        g = image[3,:,:]
                        r = image[4,:,:]
                        image = torch.stack((r,g,b), dim=0)
                        mean = list(reversed(configs["mean"][2:5]))
                        std = list(reversed(configs["std"][2:5]))
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
            if not yield_dataset:
                yield (image, label, sample["__key__"])
            else:
                yield (image, label, sample["__key__"], configs["dataset"])

    def get_patches_eval(src):
        for sample in src:
            image = torch.load(io.BytesIO(sample["image.pth"])).float()
            label = torch.load(io.BytesIO(sample["labels.pth"]))
            if configs['dataset']=='marida':
                image = einops.rearrange(image, "h w c-> c h w")
                label[label==-1] = 11
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
                    or configs["dataset"] == "flair2"
                    or configs["dataset"] == "bigearthnet"
                ):
                    if configs["task"] == "classification":
                        image = image.permute(1, 2, 0).numpy()
                        aug = A.Compose(
                            A.augmentations.Resize(
                                height=configs["enforce_resize"],
                                width=configs["enforce_resize"],
                                p=1.0,
                            )
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

            if configs["dataset"] == "woody":
                # Remove alpha channel from combined uav datasets (Some have it some don't)
                image = image[:3, :, :]
            if configs["normalization"] == "minmax":
                    image /= image.max() + 1e-6
            elif configs["normalization"] == "standard":
                if "mean" not in configs or "std" not in configs:
                    print("Mean and Std not provided for this dataset. Exiting!")
                    exit(2)
                if configs['dataset'] == 'bigearthnet':
                    if configs['rgb']:
                        b = image[2,:,:]
                        g = image[3,:,:]
                        r = image[4,:,:]
                        image = torch.stack((r,g,b), dim=0)
                        mean = list(reversed(configs["mean"][2:5]))
                        std = list(reversed(configs["std"][2:5]))
                        normalization = transforms.Normalize(mean=mean, std=std)
                    elif configs['sar']:
                        image = image[:2,:,:]
                        mean = list((configs["mean"][:2]))
                        std = list((configs["std"][:2]))
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
    max_train_index = max_train_shard.split("-train-")[-1][:-4]

    """if "dataset_percentage" in configs and configs["dataset_percentage"] is not None:
        max_train_index_int = int(int(max_train_index) * configs["dataset_percentage"])
        max_train_index = "{:06d}".format(max_train_index_int)"""

    train_shards = os.path.join(
        configs["webdataset_path"],
        "train",
        "sample-train-{000000.." + max_train_index + "}.tar",
    )
    if "dataset_percentage" in configs and configs["dataset_percentage"] is not None:
        samples = []
        for i in range(int(max_train_index) + 1):
            sh = "{:06d}".format(i)
            samples.append(os.path.join(configs["webdataset_path"], "train", "sample-train-" + sh + ".tar"))
        # Shuffle samples with seed 222
        random.Random(222).shuffle(samples)
        train_shards = samples[: int(len(samples) * configs["dataset_percentage"])]

    if not self_supervised and not supervised_foundation:
        train_dataset = wds.WebDataset(train_shards, shardshuffle=True, resampled=False).shuffle(
            configs["webdataset_shuffle_size"]
        )
        train_dataset = train_dataset.compose(get_patches)
        train_dataset = train_dataset.batched(configs["batch_size"], partial=False)
    elif supervised_foundation:
        train_dataset = wds.WebDataset(train_shards, shardshuffle=True, resampled=resample_shards).shuffle(
            configs["webdataset_shuffle_size"]
        )
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
                wds.shuffle(configs["webdataset_shuffle_size"]),
                get_patches,
                wds.batched(configs["batch_size"], partial=False),
            )
        else:
            train_dataset = wds.WebDataset(full_shards, shardshuffle=True, resampled=False).shuffle(
                configs["webdataset_shuffle_size"]
            )
            train_dataset = train_dataset.compose(get_patches)
            train_dataset = train_dataset.batched(configs["batch_size"], partial=False)

    train_loader = wds.WebLoader(
        train_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=configs["pin_memory"],
        prefetch_factor=configs["prefetch_factor"],
        persistent_workers=configs["persistent_workers"],
    )
    train_loader = (
        train_loader.unbatched()
        .shuffle(
            configs["webdataset_shuffle_size"],
            initial=configs["webdataset_initial_buffer"],
        )
        .batched(configs["batch_size"])
    )
    if repeat:
        train_loader = train_loader.repeat()

    max_val_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "val", "*.tar")))[-1]
    max_val_index = max_val_shard.split("-val-")[-1][:-4]
    val_shards = os.path.join(
        configs["webdataset_path"],
        "val",
        "sample-val-{000000.." + max_val_index + "}.tar",
    )

    val_dataset = wds.WebDataset(val_shards, shardshuffle=False, resampled=False)
    val_dataset = val_dataset.compose(get_patches_eval)
    val_dataset = val_dataset.batched(configs["batch_size"], partial=True)

    val_loader = wds.WebLoader(
        val_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=configs["pin_memory"],
    )
    max_test_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "test", "*.tar")))[-1]
    max_test_index = max_test_shard.split("-test-")[-1][:-4]
    test_shards = os.path.join(
        configs["webdataset_path"],
        "test",
        "sample-test-{000000.." + max_test_index + "}.tar",
    )

    test_dataset = wds.WebDataset(test_shards, shardshuffle=False, resampled=False)
    test_dataset = test_dataset.compose(get_patches_eval)
    test_dataset = test_dataset.batched(configs["batch_size"], partial=True)

    test_loader = wds.WebLoader(
        test_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=configs["pin_memory"],
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
                    ).to(configs["device"])
                elif metric.lower() == "fscore":
                    m = F1Score(
                        task="multiclass",
                        num_classes=configs["num_classes"],
                        average=metric_aggregation_strategy,
                        multidim_average="global",
                    ).to(configs["device"])
                elif metric.lower() == "precision":
                    m = Precision(
                        task="multiclass",
                        average=metric_aggregation_strategy,
                        num_classes=configs["num_classes"],
                        multidim_average="global",
                    ).to(configs["device"])
                elif metric.lower() == "recall":
                    m = Recall(
                        task="multiclass",
                        average=metric_aggregation_strategy,
                        num_classes=configs["num_classes"],
                        multidim_average="global",
                    ).to(configs["device"])
                elif metric.lower() == "iou":
                    m = JaccardIndex(
                        task="multiclass",
                        num_classes=configs["num_classes"],
                        average=metric_aggregation_strategy,
                    ).to(configs["device"])
                else:
                    print("Metric: ", metric, " not supported!")
                    continue
            metrics.append(m)
    return tuple(metrics)


def create_loss(configs):
    if configs["loss"] == "cross_entropy":
        if configs["multilabel"]:
            return nn.BCEWithLogitsLoss()
        else:
            return nn.CrossEntropyLoss()
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
        trainer = classification.train
        tester = classification.test
    elif configs["task"] == "segmentation":
        trainer = segmentation.train
        tester = segmentation.test
    else:
        print("AAAAAA")
        print("Task: ", configs["task"], " not supported.")
        exit(3)

    return trainer, tester


# def apply_conf(orig_prediction, conf_thresh=0.3):
#     """Confidence threshold for detection evaluation"""
#     keep = orig_prediction["scores"] > conf_thresh
#     final_prediction = orig_prediction
#     final_prediction["boxes"] = final_prediction["boxes"][keep]
#     final_prediction["scores"] = final_prediction["scores"][keep]
#     final_prediction["labels"] = final_prediction["labels"][keep]
#     return final_prediction


# def apply_class_filter(orig_prediction, class_to_filter=0):
#     """Confidence threshold for detection evaluation"""
#     keep = orig_prediction["labels"] != class_to_filter
#     final_prediction = orig_prediction
#     final_prediction["boxes"] = final_prediction["boxes"][keep]
#     final_prediction["scores"] = final_prediction["scores"][keep]
#     final_prediction["labels"] = final_prediction["labels"][keep]
#     return final_prediction


def count_params(model):
    """Count trainable parameters of a PyTorch Model"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    return nb_params