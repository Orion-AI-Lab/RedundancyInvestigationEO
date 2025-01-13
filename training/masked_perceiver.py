import os
import random
import time

import einops
import numpy as np
import torch
from tqdm import tqdm
from vit_pytorch import MAE, ViT

import utilities.model_utilities as model_utils
import utilities.utils as utils
import wandb


def train_epoch(loader, mae,optimizer,epoch,configs):
    mae.train()
    num_steps_per_epoch = configs['num_steps_per_epoch']
    
    available_modalities = list(configs['modality_indices'].keys())
    modality_indices = list(range(len(available_modalities)))
    iterators = []
    for dataloader in loader:
        iterators.append(dataloader.__iter__())
    batches_per_dataset = {}
    failed_iterations = 0.0
    loss_per_modality = {}
    data_loading_time_per_dataset = {}
    reconstruction_log_data = None
    for idx in tqdm(range(num_steps_per_epoch)):
        if configs['dataset'] == 'all':
            
            modality_name, modality_index = random.choice(list(zip(available_modalities,modality_indices)))
            
            available_datasets = [d for d in configs['modality_indices'][modality_name] if d in configs['train_datasets']]

            if len(available_datasets)==0:
                failed_iterations+=1
                if failed_iterations>10:
                    print('Too many failed attempts! Validate the selected datasets and modalities!')
                    exit(3)
                continue
            else:
                failed_iterations=0

                if modality_name not in loss_per_modality:
                    # [Running loss per modality, number of iterations per modality]
                    loss_per_modality[modality_name] = [0.0,0]
            dataset_indices = list(range(len(available_datasets)))
            train_loader, dataset_name, dataset_index = random.choice(list(zip(iterators,available_datasets,dataset_indices)))
            desired_indices = configs['modality_indices'][modality_name][dataset_name]

            if dataset_name not in data_loading_time_per_dataset:
                    #[Running data loading time, Number of dataset iterations]
                    data_loading_time_per_dataset[dataset_name] = [0.0,0]
            
            if dataset_name in batches_per_dataset:
                batches_per_dataset[dataset_name] +=1
            else:
                batches_per_dataset[dataset_name] = 1
        start_time = time.time()     
        batch = train_loader.__next__()
        
        end_time = time.time()
        data_loading_time_per_dataset[dataset_name][0] += end_time - start_time
        data_loading_time_per_dataset[dataset_name][1] += 1
        if configs['mixed_precision']:
            # Creates a GradScaler once at the beginning of training.
            scaler = torch.cuda.amp.GradScaler()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=configs['mixed_precision']):

            image, _ = batch
            
            image = image[:,desired_indices,:,:]

            image = image.to(configs['device'],non_blocking=True)

            if configs['dataset']=='all':
                loss, predicted_pixels, reconstruction_data = mae((image,modality_index))
            else:
                loss, _ = mae(image)

            loss_per_modality[modality_name][0] += loss.item()
            loss_per_modality[modality_name][1] += 1
            
            if modality_name == 'aerial': #or modality_name == 'sentinel2-rgb':
                stats = configs['all_stats'][dataset_name]
                mean = [stats['mean'][i] for i in desired_indices]
                std = [stats['std'][i] for i in desired_indices]
                reconstruction_data['mean'] = mean
                reconstruction_data['std'] = std
                if reconstruction_log_data is None:
                    reconstruction_log_data = {}
                    reconstruction_log_data['patches'] = reconstruction_data['patches'][0].unsqueeze(0).cpu().clone().detach()
                    reconstruction_log_data['model_input'] = reconstruction_data['model_input'][0].unsqueeze(0).cpu().clone().detach()
                    reconstruction_log_data['masked_indices'] = reconstruction_data['masked_indices'][0].unsqueeze(0).cpu().clone().detach()
                    reconstruction_log_data['predicted'] = reconstruction_data['predicted'][0].unsqueeze(0).cpu().clone().detach()
                    reconstruction_log_data['patch_size'] = configs['patch_size']
                    reconstruction_log_data['mean'] = mean
                    reconstruction_log_data['std'] = std
                
            
            if idx%100==0:
                log_dict = {'Epoch':epoch, 'Iteration':idx,'train loss':loss.item()}
                for key in loss_per_modality:
                    log_dict[key + ' loss: '] = loss_per_modality[key][0]/loss_per_modality[key][1]
                for key in data_loading_time_per_dataset:
                    log_dict['Avg data loading time for: '+ key] = data_loading_time_per_dataset[key][0]/data_loading_time_per_dataset[key][1]
                
                if configs['wandb']:
                    if idx%configs['log_reconstruction_every']==0 and reconstruction_log_data is not None:
                        reconstructed_image = mae.reconstruct_image(patches=reconstruction_log_data['patches'],model_input=reconstruction_log_data['model_input'],
                                                                    masked_indices=reconstruction_log_data['masked_indices'],pred_pixel_values=reconstruction_log_data['predicted'],patch_size=reconstruction_log_data['patch_size'],
                                                                    mean=reconstruction_log_data['mean'],std=reconstruction_log_data['std'])[0]

                        masked_image =  mae.reconstruct_image(patches=reconstruction_log_data['patches'],model_input=reconstruction_log_data['model_input'],
                                                                    masked_indices=reconstruction_log_data['masked_indices'],patch_size=reconstruction_log_data['patch_size'],
                                                                    mean=reconstruction_log_data['mean'],std=reconstruction_log_data['std'])[0]

                        original_image = reconstruction_log_data['model_input'][0]
                        
                        reconstructed_image = einops.rearrange(reconstructed_image,'c h w -> h w c')
                        reconstructed_image /= reconstructed_image.max()
                        

                        original_image = einops.rearrange(original_image,'c h w -> h w c')
                        original_image = original_image.numpy().astype(float)
                        original_image *= np.array(reconstruction_log_data['std'])
                        original_image += np.array(reconstruction_log_data['mean'])
                        original_image /= original_image.max()
                        
                        masked_image = einops.rearrange(masked_image,'c h w -> h w c')
                        masked_image /= masked_image.max()
                        wandb_recon_image = wandb.Image(
                                reconstructed_image, 
                                caption="Reconstructed Image"
                                )
                        wandb_masked_image = wandb.Image(
                            masked_image,
                            caption="Masked Image"
                        )

                        wandb_original_image = wandb.Image(
                            original_image,
                            caption="Input Image"
                        )

                        log_dict['Input Image'] = wandb_original_image
                        log_dict['Masked Image'] = wandb_masked_image
                        log_dict['Reconstructed Image'] = wandb_recon_image
                        reconstruction_log_data = None
                    wandb.log(log_dict)
                else:
                    print(log_dict)
            
            if configs['mixed_precision']:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                    loss.backward()
                    optimizer.step() 
    print('='*20)
    print('Epoch sampling statistics')
    print(batches_per_dataset)
    print('='*20)
           

def train(configs):
    print('='*20)
    print('Initializing MAE')
    print('Training on the following modalities:')
    print(list(configs['modality_indices'].keys()))
    print('='*20)

    modality_channels = configs['modality_channels']
    int_modality_channels = {}
    for key in modality_channels:
        int_modality_channels[int(key)] = modality_channels[key]
    configs['modality_channels'] = int_modality_channels

    train_loader, _ , _ = utils.create_dataloaders(configs)
    
    # Load all stats for denormalization when logging reconstructed images
    stats_config_path = 'configs/stats/stats.json'
    stats_config = utils.load_config_file(stats_config_path)
    configs['all_stats'] = stats_config

    base_model = model_utils.create_model(configs)

    optimizer = utils.create_optimizer(configs)(base_model.parameters(),lr=configs['lr'],weight_decay=configs['weight_decay'])
    mae = torch.compile(base_model)
    mae.to(configs['device'])
    for epoch in range(configs['epochs']):
        train_epoch(train_loader,mae,optimizer,epoch,configs)
        if epoch % 10 == 0:
            torch.save(mae.state_dict(), os.path.join(configs['checkpoint_path'], 'mae_'+str(epoch)+'.pt'))
    torch.save(mae.encoder.state_dict(), os.path.join(configs['checkpoint_path'], 'mae_vit_'+str(configs['epochs'])+'.pt'))