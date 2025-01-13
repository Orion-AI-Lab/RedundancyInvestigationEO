import os

import albumentations as A
import cv2 as cv
import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyjson5 as json
import rasterio
import torch
from torchvision import transforms

import utilities


class UniversalDataset(torch.utils.data.Dataset):

    def __init__(self,datasets,configs,mode='train'):
        print('='*40)
        print('Initializing Universal Dataset - ',mode)
        print('='*40)
        pass
        

    def __len__(self):
        pass


    def __getitem__(self, index):
        
        pass
