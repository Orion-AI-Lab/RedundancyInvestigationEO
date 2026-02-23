import json
import os
import pprint
import random

# Set environment variables early (before torch import)
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'
os.environ['CUDNN_LOGINFO_DBG'] = '0'
os.environ['CUDNN_LOGERR_DBG'] = '0'
# CuBLAS deterministic backend (required for torch.bmm, matmul reproducibility on CUDA)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import utilities
import utilities.utils as utils
import argparse

# TF32: disabled during training for reproducibility (non-deterministic on Ampere+).
# Set in __main__ after configs are loaded.
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def seed_everything(seed, strict_deterministic=False):
    """Set all relevant RNG seeds. If strict_deterministic=True (MARIDA only), also enable
    cudnn.deterministic and torch.use_deterministic_algorithms; otherwise skip those to avoid
    torchmetrics device issues (e.g. AveragePrecision cumsum) on other datasets."""
    print("Setting seed to {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if strict_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("Strict deterministic mode enabled (MARIDA reproducibility).")
    else:
        torch.backends.cudnn.benchmark = True  # faster for non-MARIDA

import warnings
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")


def store_experiment_id(configs):
    json.dump(
        {"run_id": configs["wandb_run_id"]},
        open(os.path.join(configs["checkpoint_path"], "id.json"), "w"),
    )


def init_wandb(configs):
    if configs["wandb_id_resume"] is None:
        id = wandb.util.generate_id()
    else:
        id = configs["wandb_id_resume"]
    wandb.init(
        project=configs["wandb_project"],
        entity=configs["wandb_entity"],
        config=configs,
        id=id,
        resume="allow",
    )
    run = wandb.run
    name = run.name if run is not None and run.name is not None else "unnamed_run"
    configs["wandb_run_name"] = name
    configs["wandb_run_id"] = id
    if "checkpoint_path" not in configs.keys():
        checkpoint_path = utils.create_checkpoint_path(configs)
        configs["checkpoint_path"] = checkpoint_path
    store_experiment_id(configs)
    return configs


def init_offline_experiment(configs):
    configs["wandb_run_name"] = "offline"
    configs["wandb_run_id"] = "None"
    if "checkpoint_path" not in configs.keys():
        checkpoint_path = utils.create_checkpoint_path(configs)
        configs["checkpoint_path"] = checkpoint_path
    return configs


import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--method_config", default=None)
    parser.add_argument("--augmentation_config", default=None)
    parser.add_argument("--wandb_id_resume", default=None)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--mask_threshold", default=None, type=float)
    parser.add_argument("--implementation", default=None, type=str)
    parser.add_argument("--attention_mask", default=None, type=str2bool)
    parser.add_argument("--retain_percent", default=None, type=int)
    parser.add_argument("--backbone", default=None, type=str)
    parser.add_argument("--pretrained", default=None, type=str2bool)
    parser.add_argument("--apply_mask_in_infer", default=None, type=str2bool)
    #parser.add_argument("--patch_size", default=None, type=int)

    args = parser.parse_args()

    # Setup configurations
    configs = utils.load_configs(args)
    #pprint.pprint(configs)
    if args.seed is not None:
        configs["seed"] = args.seed
    strict_det = configs.get("strict_deterministic", False)
    seed_everything(configs["seed"], strict_deterministic=strict_det)

    if strict_det:
        torch.set_float32_matmul_precision("highest")
        print("strict_deterministic=True: TF32 disabled for reproducibility.")
    else:
        torch.set_float32_matmul_precision("high")

    # Setup wandb
    if configs["wandb"] and not configs["distributed"]:
        configs = init_wandb(configs)
    else:
        if configs['phase'] != 'test':
            configs = init_offline_experiment(configs)
        else:
            configs['checkpoint_path'] = configs['eval_checkpoint']

    trainer, tester = utils.create_procedures(configs)

    if configs["phase"] == "train":
        trainer(configs)
    if tester is not None:
        _, _, loader = utils.create_dataloaders(configs)
        if isinstance(loader, list):
            dataset_names = configs['dataset_names'].split(',')
            for i, sub_loader in enumerate(loader):
                tester(configs, loader=sub_loader, phase="test", dataset_name=dataset_names[i])
        else:
            tester(configs, loader=loader, phase="test")
