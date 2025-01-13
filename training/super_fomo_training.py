import pprint
import time

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

def train_epoch_spectral(loader, model, criterion, optimizer, epoch, configs, scaler):
    model.train()
    configs["num_steps_per_epoch"] = configs["num_samples_per_epoch"] // configs["batch_size"]
    num_steps_per_epoch = configs["num_steps_per_epoch"]
    modality_dictionary = {v: k for k, v in configs["modality_channels"].items()}
    if configs["distributed"]:
        num_steps_per_epoch = num_steps_per_epoch // configs["world_size"]
        device = configs["local_rank"]
        disable = not dist_utils.is_global_master(configs)
    else:
        device = configs["device"]
        disable = False

    iterators = {}
    for dataloader, dataset_name in list(zip(loader, configs["train_datasets"])):
        iterators[dataset_name] = dataloader.__iter__()

    batches_per_dataset = {}
    loss_per_dataset = {}
    data_loading_time_per_dataset = {}

    # Set up gradient accumulation
    if configs["accumulate_gradients"] is not None:
        batches_to_accumulate = configs["accumulate_gradients"]

    running_loss = 0.0
    number_of_batches = 0.0
    for idx in tqdm(range(num_steps_per_epoch), disable=disable):
        if configs["dataset"] == "all":
            # Select dataset to operate on
            available_datasets = configs["train_datasets"]
            dataset_indices = list(range(len(available_datasets)))

            # Add option for weighted dataset selection
            if configs["dataset_probabilities"] is None:
                dataset_name, dataset_index = random.choice(list(zip(available_datasets, dataset_indices)))
            else:
                choice = random.choices(
                    list(zip(available_datasets, dataset_indices)), weights=configs["dataset_probabilities"], k=1
                )
                dataset_name, dataset_index = choice[0]

            train_loader = iterators[dataset_name]

            available_modalities = configs["dataset_modality_index"][dataset_name]

            maximum_num_bands = configs['max_number_of_bands']

            if maximum_num_bands>=len(available_modalities):
                maximum_num_bands = len(available_modalities)-1
            #num_masked_channels = random.randint(0, len(available_modalities) - 1)
            #Modify selection to enforce max number of selected bands:
            num_masked_channels = random.randint(len(available_modalities) -1 - maximum_num_bands, len(available_modalities) - 1)

            # mask_channels = int(num_masked_channels*len(available_modalities))
            channels_to_select = len(available_modalities) - num_masked_channels

            desired_channels = random.sample(list(available_modalities.keys()), channels_to_select)
            desired_indices = []
            spectral_keys = []
            for modality in desired_channels:
                desired_indices.append(available_modalities[modality])
                spectral_keys.append(int(modality_dictionary[modality]))

            if dataset_name not in data_loading_time_per_dataset:
                # [Running data loading time, Number of dataset iterations]
                data_loading_time_per_dataset[dataset_name] = [0.0, 0]
                loss_per_dataset[dataset_name] = 0.0

            if dataset_name in batches_per_dataset:
                batches_per_dataset[dataset_name] += 1
            else:
                batches_per_dataset[dataset_name] = 1
        start_time = time.time()
        batch = train_loader.__next__()

        end_time = time.time()
        data_loading_time_per_dataset[dataset_name][0] += end_time - start_time
        data_loading_time_per_dataset[dataset_name][1] += 1
            
        with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
            if (
                configs["accumulate_gradients"] is None
                or (idx + 1) % batches_to_accumulate == 0
                or (idx + 1) == num_steps_per_epoch
            ):
                # we use a per iteration (instead of per epoch) lr scheduler as done in official MAE implementation
                adjust_learning_rate(optimizer, idx / num_steps_per_epoch + epoch, configs)

            image, _ = batch
            image = image[:, desired_indices, :, :]

            image = image.to(device, non_blocking=True)

            out, reconstruction_loss = model((image, spectral_keys,task))
            
            loss = criterion(out, label)
            if reconstruction_loss is not None:
                loss += configs['reconstruction_loss_weight'] * reconstruction_loss
                
        running_loss += loss.item()
        loss_per_dataset[dataset_name] += loss.item()
        number_of_batches += 1

        if idx % 100 == 0:
            if not configs["distributed"] or dist_utils.is_global_master(configs):
                log_dict = {"Epoch": epoch, "Iteration": idx, "train loss": running_loss / number_of_batches}
                running_loss = 0.0
                number_of_batches = 0.0
                for key in loss_per_dataset:
                    log_dict[key + " loss: "] = loss_per_dataset[key] / batches_per_dataset[key]
                for key in data_loading_time_per_dataset:
                    log_dict["Avg data loading time for: " + key] = (
                        data_loading_time_per_dataset[key][0] / data_loading_time_per_dataset[key][1]
                    )

                log_dict["Current Learning Rate"] = get_current_learning_rate(optimizer)

                if configs["wandb"]:
                    wandb.log(log_dict)
                else:
                    print(log_dict)

        # Scale loss according to gradient accumulation
        if configs["accumulate_gradients"] is not None:
            loss = loss / batches_to_accumulate

        if configs["mixed_precision"]:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # If gradient accumulation is enabled, update weights every batches_to_accumulate iterations.
        if (
            configs["accumulate_gradients"] is None
            or (idx + 1) % batches_to_accumulate == 0
            or (idx + 1) == num_steps_per_epoch
        ):
            if configs["mixed_precision"]:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
    print("=" * 20)
    print("Epoch sampling statistics")
    print(batches_per_dataset)
    print("=" * 20)
    
    
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
    
    # Set up gradient accumulation
    if configs["accumulate_gradients"] is not None:
        batches_to_accumulate = configs["accumulate_gradients"]
        
    for idx in tqdm.tqdm(range(max_number_iterations),disable=disable):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
            dataset_name = random.choice(list(available_datasets))
            if dataset_name in configs['cls_train_datasets']:
                task = 'classification'
            elif dataset_name in configs['segm_train_datasets']:
                task = 'segmentation'
            else:
                raise ValueError('Task not supported yet!')
            
            available_modalities = configs["dataset_modality_index"][dataset_name]

            maximum_num_bands = configs['max_number_of_bands']

            if maximum_num_bands>=len(available_modalities):
                maximum_num_bands = len(available_modalities)-1
            #num_masked_channels = random.randint(0, len(available_modalities) - 1)
            #Modify selection to enforce max number of selected bands:
            if task == 'classification':
                num_masked_channels = random.randint(len(available_modalities) -1 - maximum_num_bands, len(available_modalities) - 1)
            else:
                num_masked_channels = maximum_num_bands
                
            # mask_channels = int(num_masked_channels*len(available_modalities))
            channels_to_select = len(available_modalities) - num_masked_channels

            desired_channels = random.sample(list(available_modalities.keys()), channels_to_select)
            desired_indices = []
            spectral_keys = []
            for modality in desired_channels:
                desired_indices.append(available_modalities[modality])
                spectral_keys.append(int(modality_dictionary[modality]))

            
            train_loader = iterators[dataset_name]
            
            
            batch = train_loader.__next__()
            image, label, _ = batch
            
            image = image[:, desired_indices, :, :]
            image = image.to(device)
            label = label.to(device)
            

            out, reconstruction_loss = model((image, spectral_keys,task))
            
            loss = criterion(out, label)
            if reconstruction_loss is not None:
                loss += configs['reconstruction_loss_weight'] * reconstruction_loss
                
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

    for key in base_model.classification_fc:
        base_model.classification_fc[key].to(configs["device"])
    for key in base_model.segmentation_fc:
        base_model.segmentation_fc[key].to(configs["device"])
    
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
        torch.save(model.module, os.path.join(configs["checkpoint_path"], "super_foundation_" + str(epoch) + ".pt"))
        state_dict = {"base_model": model.module.base_model.state_dict()}
        state_dict["fc"] = {}
        for key in model.module.fc:
            state_dict["fc"][key] = model.module.fc[key].state_dict()
        torch.save(
            state_dict, os.path.join(configs["checkpoint_path"], "super_foundation_weights_" + str(epoch) + ".pt")
        )
    if dist_utils.is_global_master(configs):
        wandb.finish()