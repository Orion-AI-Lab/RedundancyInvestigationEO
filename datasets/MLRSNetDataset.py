import os
import pickle
import pprint
import random
import warnings

import cv2 as cv
import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyjson5 as json
import rasterio
from rasterio import features
import torch
from tqdm import tqdm

import albumentations as A

warnings.simplefilter("ignore")

class MLRSNetDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode='train'):
        self.configs = configs
        self.root_path = os.path.join(configs['root_path'], 'MLRSNet')
        self.mode = mode
        self.annotation_files = os.listdir(os.path.join(self.root_path, "labels")) 
        self.num_labels = len(self.annotation_files)
        self.label_keys = None
        self.samples = []
        
        for annotation_file in self.annotation_files:
            if annotation_file.endswith(".csv"):
                annotation = pd.read_csv(os.path.join(self.root_path, "labels", annotation_file))
                if self.label_keys is None:
                    self.label_keys = annotation.columns.tolist()[1:]
                for index, row in annotation.iterrows():
                    sample = {}
                    sample['label'] = row.tolist()[1:]
                    
                    # Extract the image filename from CSV
                    img_filename = row['image']
                    
                    # Extract category from filename
                    # Pattern: category_number.jpg (e.g., "wetland_00001.jpg" -> "wetland")
                    # Split by underscore and take all parts except the last one (which is the number)
                    parts = img_filename.replace('.jpg', '').split('_')
                    category = '_'.join(parts[:-1])  # Join all parts except the number
                    
                    # Construct full path: Images/category/filename
                    sample['path'] = os.path.join(self.root_path, "Images", category, img_filename)
                    
                    if self.mode == "val":
                        if "2.jpg" in sample['path']:               
                            self.samples.append(sample)
                    elif self.mode == "test":
                        if "3.jpg" in sample['path']:               
                            self.samples.append(sample)
                    else:
                        if "2.jpg" not in sample['path'] and "3.jpg" not in sample['path']:
                            self.samples.append(sample)
        
        # Shuffle samples
        random.Random(999).shuffle(self.samples)
        self.num_examples = len(self.samples)
        
        # Quick validation check (optional, can be removed for faster loading)
        if self.num_examples > 0:
            # Check first few samples
            missing_count = 0
            for i in range(min(10, self.num_examples)):
                if not os.path.exists(self.samples[i]['path']):
                    missing_count += 1
                    print(f"WARNING: Path doesn't exist: {self.samples[i]['path']}")
            
            if missing_count > 0:
                print(f"WARNING: {missing_count} out of first 10 samples have missing paths!")
        
        print(f"Loaded {self.num_examples} samples for {mode} mode")
    
    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        sample = self.samples[index]
        
        image = cv.imread(sample['path'])
        
        if image is None:
            raise ValueError(f"Failed to load image: {sample['path']}")
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        if image.shape[0] != 256 or image.shape[1] != 256:
            image = cv.resize(image, (256, 256))
        
        image = einops.rearrange(image, 'h w c -> c h w')
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32)
        
        label = torch.from_numpy(np.asarray(sample['label']))
        return image, label