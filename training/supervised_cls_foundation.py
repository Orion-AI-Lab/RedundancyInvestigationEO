import pprint

import numpy as np
import pyjson5 as json
import torch
import tqdm
import wandb
import os
import utilities.model_utilities as model_utils
import utilities.utils as utils
from utilities.model_utilities import adjust_learning_rate, get_current_learning_rate
import random
import torch.nn as nn
from utilities import distributed_utils as dist_utils


def train_epoch(loader, model, optimizer, criterion, epoch, configs, scaler):
    if not configs["linear_evaluation"]:
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
    
    max_number_iterations = configs["max_number_iterations"] // configs["batch_size"]

    if configs['distributed']:
        max_number_iterations = max_number_iterations // configs['world_size']
        device = configs['local_rank']
        disable = not dist_utils.is_global_master(configs)
    else:
        device = configs['device']
        disable = False
    
    iterators = {}
    for dataloader, dataset_name in list(zip(loader, configs["train_datasets"])):
        iterators[dataset_name] = dataloader.__iter__()
    available_datasets = configs["train_datasets"]
    for idx in tqdm.tqdm(range(max_number_iterations),disable=disable):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
            dataset_name = random.choice(list(available_datasets))
            indices = list(configs["dataset_modality_index"][dataset_name].values())
            train_loader = iterators[dataset_name]
            batch = train_loader.__next__()
            image, label, dataset = batch
            print(dataset_name)
            image = image[:, indices, :, :]
            image = image.to(device)
            label = label.to(device)

            out = model(image, dataset_name)
            loss = criterion(out, label)
        if idx % 100 == 0:
            if not configs['distributed'] or dist_utils.is_global_master(configs):
                log_dict = {"Epoch": epoch, "Iteration": idx, "train loss": loss.item()}
                if configs["schedule"] != "none":
                    log_dict["Current Learning Rate"] = get_current_learning_rate(optimizer)

                if configs["wandb"]:
                    wandb.log(log_dict)
                else:
                    print(log_dict)
        if configs["mixed_precision"]:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if configs["schedule"] == "cos":
            if "num_samples" in configs:
                num_steps = configs["num_samples"] // configs["batch_size"]
            else:
                num_steps = 10000
            adjust_learning_rate(optimizer, idx / num_steps + epoch, configs)


def train(configs):
    print("=" * 20)
    print("Initializing supervised classification foundation model trainer")
    print("=" * 20)
    criterion = utils.create_loss(configs)
    base_model = model_utils.create_model(configs)
    
    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    #Setup distributed mode
    if configs['seed'] is not None:
        dist_utils.seed(configs)
        
    if configs['distributed']:
        configs = dist_utils.init_distributed(configs)
        print('setting device for rank: ',configs['rank'])
        print('setting device for local rank: ',configs['local_rank'])
        print('Current device: ',torch.cuda.current_device())
        print('Total count: ',torch.cuda.device_count())
        print('world size ',configs['world_size'])

        torch.cuda.set_device(configs["local_rank"])
        
        if dist_utils.is_global_master(configs):
            id = wandb.util.generate_id()
            configs['wandb_id'] = id
            wandb.init(project=configs['wandb_project'], entity=configs['wandb_entity'], config=configs, id=id, resume="allow")
    
    if not configs['distributed']:
        world_size = 1
    else:
        world_size = configs['world_size']
    
    configs['lr'] = configs['lr'] * world_size
    
    optimizer = utils.create_optimizer(configs)(
        base_model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"]
    )

    # compile model (torch 2.0)
    # model = torch.compile(base_model)

    train_loader, _, _ = utils.create_dataloaders(configs)

    for key in base_model.fc:
        base_model.fc[key].to(configs["device"])
    
    if not configs['distributed']:
        base_model.to(configs['device'])
    else:
        base_model.to(configs['local_rank'])
        print('='*20)
        print('Initializing model for local_rank: ',configs['local_rank'])
        print('='*20)
        base_model = torch.nn.parallel.DistributedDataParallel(
            base_model, device_ids=[configs["local_rank"]],output_device=configs['local_rank'],find_unused_parameters=True
        )
    if not configs['distributed'] or dist_utils.is_global_master(configs):
        if configs['wandb']:
            wandb.watch(base_model)

    best_loss = 10000.0
    if configs["start_epoch"] is None:
        start_epoch = 0
    else:
        start_epoch = configs["start_epoch"]

    model = base_model
    for epoch in range(start_epoch, configs["epochs"]):
        train_epoch(train_loader, model, optimizer, criterion, epoch, configs, scaler)
        torch.save(model.module, os.path.join(configs["checkpoint_path"], "supervised_foundation_" + str(epoch) + ".pt"))
        state_dict = {"base_model": model.module.base_model.state_dict()}
        state_dict["fc"] = {}
        for key in model.module.fc:
            state_dict["fc"][key] = model.module.fc[key].state_dict()
        torch.save(
            state_dict, os.path.join(configs["checkpoint_path"], "supervised_foundation_weights_" + str(epoch) + ".pt")
        )
    if dist_utils.is_global_master(configs):
        wandb.finish()