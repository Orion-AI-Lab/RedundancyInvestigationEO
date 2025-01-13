import os
import random
from pathlib import Path

import albumentations as A
import cv2 as cv
import einops
import geopandas as gpd
import kornia
import matplotlib.pyplot as plt
import numpy as np
import pyjson5 as json
import rasterio
import rasterio.mask
import torch
from rasterio import features
from rasterio.windows import Window
from tqdm import tqdm
from torchvision import transforms

import utilities

'''
Data loading for Spekboom canopy detection dataset published in:
Galuszynski, N. C., Duker, R., Potts, A. J., & Kattenborn, T. (2022). Automated mapping of Portulacaria afra canopies for restoration
monitoring with convolutional neural networks and heterogeneous unmanned aerial vehicle imagery. PeerJ, 10, e14219.
'''

class SpekboomDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode = 'train'):
        print('='*40)
        print('Initializing Spekboom dataset - mode: ',mode)
        print('='*40)
        self.configs = configs
        self.root_path = os.path.join(configs['root_path'],'Spekboom','data_spekboom')
        self.mode = mode        
        self.samples = []
        self.full_samples = []
        self.test_captions = ['P0011 - M2P','P0150 - P4','Y0093-M2P']
        areas = os.listdir(self.root_path)
        if self.configs['augment']:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None

        if self.configs['normalization']=='standard':
            self.normalization = transforms.Normalize(mean=self.configs['mean'],std=self.configs['std'])
        
        for area in areas:
            shape_path = os.path.join(self.root_path,area,'Shapefiles')
            aoi_path = os.path.join(shape_path,area[:5]+'_AOI.shp')
            shape = os.path.join(shape_path,area[:5]+'_Spekboom.shp')            
            image_path = os.path.join(self.root_path,area,'Stitched')
            if os.path.exists(image_path):
                files = os.listdir(image_path)
            else:
                continue
            tif_file = None
            for file in files:
                if file.endswith('.tif'):
                    tif_file = file
            if tif_file is None:
                continue
            tif_file = os.path.join(image_path,tif_file)
            tiles_path = os.path.join(image_path,'tiles_'+str(configs['tile_size']))
            masks_path = os.path.join(image_path,'masks_'+str(configs['tile_size']))
            
            if area in self.test_captions and self.mode!='test':
                    print(area,' Belongs to test set')
                    continue
            if self.mode=='test' and area not in self.test_captions:
                continue
            
            if (os.path.exists(tiles_path) and os.path.exists(masks_path)) and configs['tiling']==False:
                print('Tiles for area: ',area)
                print('Skipping tiling.')
                tiles = os.listdir(tiles_path)
                for tile in tiles:
                    id = tile.split('_')[1:]
                    id = '_'.join(id)
                    sample = {'image':os.path.join(tiles_path,'tile_'+id),'mask':os.path.join(masks_path,'tile_'+id)}
                    self.samples.append(sample)
            else:
                Path(tiles_path).mkdir(parents=True, exist_ok=True)
                Path(masks_path).mkdir(parents=True, exist_ok=True)
                self.samples.extend(self.tilerize(tif_file,shape,aoi_path,tiles_path,masks_path))

            sample = {'image':tif_file,'aoi_path':aoi_path,'shapefile':shape}
            self.full_samples.append(sample)
        
        random.Random(999).shuffle(self.samples)
        if mode=='train':
            self.samples = self.samples[:int(0.9*len(self.samples))]
        elif mode=='val':
            self.samples = self.samples[int(0.9*len(self.samples)):]
        elif mode!='test':
            print('Mode: ',mode, ' not supported!')
            exit(2)

        self.num_examples = len(self.samples)    
        print('Mode: ',mode, ' Number of examples: ',self.num_examples)
    def __len__(self):
        return self.num_examples


    def plot(self,index=0):
        sample = self.full_samples[index]
        aoi_path = sample['aoi_path']
        shape = sample['shapefile']
        tif_file = sample['image']
        aoi_shape = gpd.read_file(aoi_path)
        shapefile = gpd.read_file(shape)
        aoi_shape =  aoi_shape.dropna(subset=['geometry'])
        with rasterio.open(tif_file) as src:
            metadata = src.profile
            
            image, out_transform = rasterio.mask.mask(src, aoi_shape.geometry, crop=True)
            metadata['transform']=out_transform
            metadata['height']=image.shape[1]
            metadata['width']=image.shape[2]

            image = einops.rearrange(image,'c h w -> h w c')
            
            rasterized = np.zeros((metadata['height'], metadata['width']), dtype=metadata['dtype'])
            shapes = ((geom, 1) for geom in shapefile.geometry)
            mask = features.rasterize(shapes=shapes, fill=0, out=rasterized, transform=metadata['transform'])

        _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        ax[0].imshow(mask)
        ax[0].set_title('Mask')

        ax[1].imshow(image/image.max())
        ax[1].set_title(tif_file.split('/')[-1][:-4])
        plt.savefig('Spekboom_sample_'+str(index)+'.png')
        plt.show()


    def tilerize(self,tif_path,shape_path,aoi_path,tiles_path,masks_path):
        print('Initialize tile creation with shapefile paths:')
        print('Mask shapefile: ',shape_path)
        print('AOI shapefile: ',aoi_path)
        if not os.path.exists(shape_path):
            print('Shapefile does not exist: ',shape_path)
        if not os.path.exists(aoi_path):
            print('Shapefile does not exist: ',aoi_path)
        shapefile = gpd.read_file(shape_path)
        aoi_shape = gpd.read_file(aoi_path)
        aoi_shape =  aoi_shape.dropna(subset=['geometry'])

        #Crop to AOI
        with rasterio.open(tif_path) as src:
            metadata = src.profile
            
            tif, out_transform = rasterio.mask.mask(src, aoi_shape.geometry, crop=True)
            metadata['transform']=out_transform
            metadata['height']=tif.shape[1]
            metadata['width']=tif.shape[2]

            rasterized = np.zeros((metadata['height'], metadata['width']), dtype=metadata['dtype'])
            shapes = ((geom, 1) for geom in shapefile.geometry)
            mask = features.rasterize(shapes=shapes, fill=0, out=rasterized, transform=metadata['transform'])
        
        num_rows = tif.shape[1]
        num_cols = tif.shape[2]
        tile_size = self.configs['tile_size']
        samples = []
        print('Full area size: ', mask.shape)
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
    
    #TODO define preprocess pipeline e.g Normalization etc.
    def __getitem__(self, index):

        sample = self.samples[index]
        mask_path = sample['mask']
        image_path = sample['image']
        with rasterio.open(image_path) as src:
                image = src.read()

        with rasterio.open(mask_path) as src2:
            mask = src2.read()
        
        image = np.clip(image, 0, np.finfo(np.float32).max)
        image = image.astype('float32')
        if np.isinf(image).any():
            print('Image is inf: ')
            print(image)
            exit(3)

        #Is this the correct condition?
        if mask.shape[0]!=self.configs['tile_size'] or mask.shape[1]!=self.configs['tile_size']:
            image = einops.rearrange(image,'c h w -> h w c')
            resize =  A.Compose([A.augmentations.Resize(height=self.configs['tile_size'],width=self.configs['tile_size'],p=1.0)])
             
            transform = resize(image=image,mask=mask.squeeze())
            image = transform['image']
            mask = transform['mask']
            image = einops.rearrange(image,'h w c -> c h w')
        
        if not self.configs['webdataset']:
            if self.configs['augment'] and self.mode=='train':
                image = einops.rearrange(image,'c h w -> h w c')
                transform = self.augmentations(image=image,mask=mask.squeeze())
                image = transform['image']
                mask = transform['mask']
                image = einops.rearrange(image,'h w c -> c h w')
            if self.configs['normalization']=='minmax':
                image /=  (image.max() + 1e-5)
            elif self.configs['normalization']=='standard':
                image = torch.from_numpy(image).float()
                image = self.normalization(image)
            else:
                image = torch.from_numpy(image).float()
        else:
            image = torch.from_numpy(image).float()

        mask = torch.from_numpy(mask).long()
        return image, mask