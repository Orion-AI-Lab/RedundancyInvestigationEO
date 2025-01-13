import os
import pprint
import random

import cv2 as cv
import einops
import kornia
import matplotlib.pyplot as plt
import numpy as np
import pyjson5 as json
import rasterio
import torch
from tqdm import tqdm
from torchvision import transforms
import pyjson5 as json
import glob
import utilities

'''
Data loading for the GLAD Dataset
'''

class GLADDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode = 'train'):
        self.configs = configs
        self.root_path = os.path.join(configs['root_path'],'GLAD')
        self.mode = mode
        if self.configs['augment']:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None        

        if self.configs['normalization']=='standard':
            self.normalization = transforms.Normalize(mean=self.configs['mean'],std=self.configs['std'])
        
        self.test_tiles = ["30N_090E","30N_100E", "20N_090E","20N_100E"]

        first_images = os.path.join(self.root_path,'first')

        image_list = os.listdir(first_images)
        self.samples = []
        self.full_samples = []
        for image in image_list:
            image_tile = image.split('first_')[-1][:-4]
            if image_tile in self.test_tiles and self.mode != 'test':
                continue
            sample = {}
            sample['first_path'] = os.path.join(first_images,image)
            sample['last_path'] = sample['first_path'].replace('first','last')
            sample['data_mask'] = sample['first_path'].replace('first','datamask')
            sample['lossyear'] = sample['first_path'].replace('first','lossyear')
            sample['gain'] = sample['first_path'].replace('first','gain')
            sample['treecover'] = sample['first_path'].replace('first','treecover')
            self.full_samples.append(sample)
        self.num_examples = len(self.full_samples)
        print('Mode: ',mode,' - Number of examples: ',self.num_examples)
        exit(2)
    '''
    TODO crop tifs to tiles
    '''
    def tilerize(self,first_tif_path,last_tif_path,gain_path,lossyear_path,treecover_path):
        print('Initialize tile creation with paths:')
        print('Treecover path: ',treecover_path)
        print('First tif path: ',first_tif_path)
       
        
        with rasterio.open(first_tif_path) as src:
            metadata = src.profile
            tif = src.read()
            print(tif.shape)
            
        
        num_rows = tif.shape[1]
        num_cols = tif.shape[2]
        tile_size = self.configs['tile_size']
        samples = []
        print('Full area size: ', tif.shape)
        print('Desired tile size: ',tile_size)

        print('Saving tiles')

        for row in tqdm(range(0,num_rows,tile_size)):
            for col in tqdm(range(0,num_cols,tile_size)):
                window = rasterio.windows.Window(col, row, tile_size, tile_size)

                tile = tif[:, window.row_off:window.row_off+window.height, window.col_off:window.col_off+window.width]
                tile_profile = metadata.copy()
                tile_profile.update({
                    'height': tile.shape[1],
                    'width': tile.shape[2],
                    'transform': rasterio.windows.transform(window, metadata['transform'])
                })

                mask_tile = mask[window.row_off:window.row_off+window.height, window.col_off:window.col_off+window.width]
                mask_tile_profile = metadata.copy()
                
                mask_tile_profile.update({
                    'height': mask_tile.shape[0],
                    'width': mask_tile.shape[1],
                    'transform': rasterio.windows.transform(window, metadata['transform'])
                })
                
                with rasterio.open(os.path.join(tiles_path,f'tile_{row}_{col}.tif'), 'w', **tile_profile) as dst:
                    dst.write(tile)
                
                mask_tile_profile['count']=1
                with rasterio.open(os.path.join(masks_path,f'tile_{row}_{col}.tif'), 'w', **mask_tile_profile) as mask_dst:
                    mask_dst.write(mask_tile,indexes=1)
                sample = {'image':os.path.join(tiles_path,f'tile_{row}_{col}.tif'),'mask':os.path.join(masks_path,f'tile_{row}_{col}.tif')}
                samples.append(sample)
        return samples
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self,index):
        pass