import glob
import os
import pickle
import random

import albumentations as A
import cv2 as cv
import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyjson5 as json
import rasterio
import torch
import tqdm
from torchvision import transforms
import rioxarray as rio

class TallosDataset(torch.utils.data.Dataset):

    def __init__(self,configs,mode='train'):
        print('='*40)
        print('Initializing Tallos mode - ',mode)
        print('='*40)

        self.configs = configs
        self.root_path = os.path.join(configs['root_path'],'TalloS')
        self.sar_root_path = os.path.join(configs['root_path'],'TalloS_SAR')
        self.dem_root_path = configs['dem_path']
        self.mode = mode
        self.split_path = configs['split_path']
        self.splits = json.load(open(self.split_path,'r'))
        
        self.records = pd.read_csv(os.path.join(self.root_path,'annotations.csv'))
        unique_labels = []
        for idx, row in self.records.iterrows():
            unique_labels.extend(eval(row[self.configs['class_level']]))
        self.total_labels = np.unique(unique_labels)
        self.label_map = {}
        for idx,label in enumerate(self.total_labels):
            self.label_map[label] = idx
        
        print('='*20)
        print('Class Level: ',configs['class_level'])
        print('Number of classes: ',len(self.label_map))
        print('='*20)
        self.samples = []
        if not os.path.isfile(os.path.join(self.root_path,mode+'_samples.json')):

            for idx, row in self.records.iterrows():
                for year in range(2017,2024):
                    lon = row['longitude']
                    lat = row['latitude']
                    for cell in self.splits[mode]:
                        min_lon = cell[0][0]
                        min_lat = cell[0][1]
                        max_lon = cell[1][0]
                        max_lat = cell[1][1]
                        
                        #Check if location belongs in this set
                        if lon>=min_lon and lon<max_lon and lat>=min_lat and lat<max_lat:
                            try:
                                products = eval(row[str(year)])
                            except:
                                print(row[str(year)])
                                continue
                            for product in products:
                                sample_path =  os.path.join(self.root_path,product[0],str(lon) + '_' + str(lat))
                                sar_sample_path = os.path.join(self.sar_root_path,product[0],str(lon) + '_' + str(lat))
                                dem_path = os.path.join(self.dem_root_path,product[0],str(lon) + '_' + str(lat),"DEM.tif") #os.path.join(sample_path,'DEM.tif')
                                sentinel2_path = os.path.join(sample_path,'data.tif')
                                sar_path = os.path.join(sar_sample_path,'sar_data.tif')
                                if not os.path.isfile(sar_path):
                                    print(sar_path)
                                    continue
                                with rasterio.open(sentinel2_path) as src:
                                    if src.count==10:
                                        sample = {'sentinel2':sentinel2_path,'sar':sar_path,'dem':dem_path, 'label':self.get_label(sample_path)}
                                        self.samples.append(sample)
            json.dump({mode:self.samples},open(os.path.join(self.root_path,mode+'_samples.json'),'wb')) 
        else:
            print('='*20)
            print('Loading samples from checkpoint - mode: ',mode)
            print('='*20)
            self.samples = json.load(open(os.path.join(self.root_path,mode+'_samples.json'),'rb'))[mode]
        if self.configs['normalization']=='standard':
            self.normalization = transforms.Normalize(mean=self.configs['mean'],std=self.configs['std'])
        
        self.num_examples = len(self.samples)
        print('Mode: ',mode,' Number of examples: ',self.num_examples)
    
    def __len__(self):
        return self.num_examples

    def plot(self,index=0):
        pass
    
    def get_label(self,path):
        coords_path = os.path.join(path,'overlapping_coordinates.json')
        overlapping_coords = list(json.load(open(coords_path,'r')).values())[0]
        label = np.zeros(self.total_labels.shape[0])
        
        for coordinate in overlapping_coords:
            coordinate = (float(coordinate[0]),float(coordinate[1]))
            coordinate = np.asarray(coordinate).round(5)
            record = self.records[(self.records['longitude']==(coordinate[0]))&(self.records['latitude']==(coordinate[1]))]
        
            unique_labels = []
            for idx, row in record.iterrows():
                unique_labels.extend(eval(row[self.configs['class_level']]))
            labels = np.unique(unique_labels)
            labels = [x for x in labels if isinstance(x,str)]
            for l in labels:
                label[self.label_map[l]] = 1

        return label

    def __getitem__(self,index):
        
        sample = self.samples[index]
        if self.configs['input_mode']=='single_image':
            sentinel_2 = rio.open_rasterio(sample['sentinel2'])
            label = sample['label']
            sentinel_2 = sentinel_2.to_numpy()
            sar = rio.open_rasterio(sample['sar']).to_numpy()
            dem = rio.open_rasterio(sample['dem'])

            if dem.max().data.astype(np.float32) == dem.rio.nodata.astype(np.float32) or (dem.isnull()).any():
                exit(2)
            else:
                nan_flag=0
                dem = dem.to_numpy()

            image = np.vstack([sentinel_2,sar,dem]).astype('float')
            

            if (image.shape[1],image.shape[2])!=(224,224):
                image = einops.rearrange(image,'c h w -> h w c')
                resize =  A.Compose([A.augmentations.Resize(height=224,width=224,p=1.0)])
                transform = resize(image=image)
                image = transform['image']
                image = einops.rearrange(image,'h w c -> c h w')

            if not self.configs['webdataset']:
                if self.configs['augment']:
                    image = einops.rearrange(image, 'c h w -> h w c')
                    transform = self.augmentations(image=image)
                    image = transform['image']
                    image = einops.rearrange(image, 'h w c -> c h w')
                if self.configs['normalization']=='minmax':
                    s1 = torch.from_numpy(image[:2,:,:])/25 +1
                    s2 = torch.from_numpy(image[2:,:,:])/10000
                    image = torch.cat((s1,s2),dim=0)
                elif self.configs['normalization']=='standard':
                    image = torch.from_numpy(image).float()
                    image = self.normalization(image)
                else:
                    image = torch.from_numpy(image).float()
            else:
                    image = torch.from_numpy(image).float()
            
            if image is None:
                print('Image is none: ',sample['sentinel2'])
            if label is None:
                print('Image is None: ',sample['sentinel2'])
            return image, torch.from_numpy(np.asarray(label))
            
        else:
            pass