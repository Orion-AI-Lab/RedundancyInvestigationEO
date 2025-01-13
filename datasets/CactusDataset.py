import os
import pprint
import random

import cv2 as cv
import kornia
import numpy as np
import pyjson5 as json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import utilities
import albumentations as A
from torchvision import transforms

'''
Data loading for Columnar Cactus Dataset published in:
E. Lpez-Jimnez, J. I. Vasquez-Gomez, M. A. Sanchez-Acevedo, J. C. Herrera-Lozada, and A. V.
Uriarte-Arcia. Columnar cactus recognition in aerial images using a deep learning approach.
Ecological Informatics, 2019.
'''

class CactusDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode = 'train'):
        print('='*40)
        print('Initializing Cactus dataset - mode: ',mode)
        print('='*40)
        self.configs = configs
        self.root_path = os.path.join(configs['root_path'],'Cactus')
        self.mode = mode
        if self.configs['augment']:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None        
        if self.configs['normalization']=='standard':
            self.normalization = transforms.Normalize(mean=self.configs['mean'],std=self.configs['std'])

        self.training_set_path = os.path.join(self.root_path,'training_set')
        self.validation_set_path = os.path.join(self.root_path,'validation_set')
        self.samples = []

        if self.mode == 'train' or self.mode == 'val':
            path = self.training_set_path

        elif self.mode == 'test':
            path = self.validation_set_path
        else:
            print('Mode not supported')
            exit(2)
        
        classes = os.listdir(path)
        for class_ in classes:
            if 'no_cactus' in class_:
                label = 0
            else:
                label = 1
            samples_path = os.path.join(path,class_)
            files = os.listdir(samples_path)
            for file in files:
                file_path = os.path.join(samples_path,file)
                sample = {'path':file_path,'label':label}
                self.samples.append(sample)
        
        if mode=='val':
            random.Random(999).shuffle(self.samples)
            self.samples = self.samples[int(0.9*len(self.samples)):]
        elif mode=='train':
            random.Random(999).shuffle(self.samples)
            self.samples = self.samples[:int(0.9*len(self.samples))]
        self.num_examples = len(self.samples)
        print('Number of samples: ',self.num_examples)

    def __len__(self):
        return self.num_examples

    def plot(self,index=0):
        sample = self.samples[index]
        image = cv.imread(sample['path'])
        label = sample['label']
        plt.imshow(image)
        if label==1:
            title='Cactus'
        else:
            title='No cactus'
        plt.title('Label: '+title)
        plt.savefig('sample_'+str(index)+'.png')
        plt.show()

    def __getitem__(self, index):
        #Toy preprocessing pipeline for dev purposes
        sample = self.samples[index]
        image = cv.imread(sample['path'])
        label = sample['label']
        image = image.astype('float')
        if not self.configs['webdataset']:
            resize =  A.Compose([A.augmentations.Resize(height=120,width=120,p=1.0)])(image=image)
            image = resize['image']
            if self.configs['augment']:
                transform = self.augmentations(image=image)
                image = transform['image']
            image = torch.from_numpy(image).permute(2,0,1).float()
            if self.configs['normalization']=='minmax':
                image /= 255
            elif self.configs['normalization']=='standard':
                image = self.normalization(image)
        else:
            image = torch.from_numpy(image).permute(2,0,1).float()
        return image, label

