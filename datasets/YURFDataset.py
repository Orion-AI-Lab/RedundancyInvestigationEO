import os
import pprint
import random

import cv2 as cv
import numpy as np
import pyjson5 as json
import torch
from tqdm import tqdm

'''
Data loading class for the tree classification dataset published in:
Kentsch, Sarah, et al. "Computer vision and deep learning techniques for the analysis of drone-acquired forest images,
a transfer learning study." Remote Sensing 12.8 (2020): 1287.
'''


class YurfDataset(torch.utils.data.Dataset):
    def __init__(self, configs,mode='train'):
        print('='*20)
        print('Initializing YURF dataset - mode: ',mode)
        print('='*20)
        self.configs = configs
        self.root_path = configs['root_path']
        self.root_path = os.path.join(self.root_path,'YURF_Shonai')
        self.shonai_path = os.path.join(self.root_path,'Coastal_Forest_Data')
        self.yurf_path = os.path.join(self.root_path,'YURF')
        self.area = configs['area']
        if self.configs['area'] == 'all':
            self.area = ["yurf","shonai_coastal_forest"]
        elif self.configs['area'] == 'yurf':
            self.area = ["yurf"]
        elif self.configs['area'] == 'shonai_coastal_forest':
            self.area = ["shonai_coastal_forest"]
        else:
            print('Uknown area')
            exit(3)
        
        if "shonai_coastal_forest" in self.area:
            print('Loading paths for the Shonai coastal forest')
            shonai_orthomosaics = self.collect_shonai_orthomosaics()    

        if "yurf" in self.area:
            print('Loading paths for the YURF')
            yurf_orthomosaics = self.collect_yurf_orthomosaics()
        
    def collect_yurf_orthomosaics(self):
                
        site_1 = {"orthomosaic":["36_Orthomosaic_site1.tif","39_Orthomosaic_Site1.tif","42_orthomosaic_Site1.tif"],"annotation":["wM3","wM4","wM5"]}
        site_2 = {"orthomosaic":["37_Orthomosaic_Site2.tif"],"annotation":["wM2"]}
        site_3 = {"orthomosaic":["38_Orthomosaic_Site3.tif"],"annotation":["wM1"]}
        site_4 = {"orthomosaic":["40_Orthomosaic_site4.tif"],"annotation":["wM7"]}
        site_6 = {"orthomosaic":["41_orthomosaic_Site6.tif"],"annotation":["wM6"]}

        sites = {"site_1":site_1,"site_2":site_2,"site_3":site_3,"site_4":site_4,"site_5":site_6}
        for key in sites:
            for idx, annotation in enumerate(sites[key]['annotation']):
                sites[key]['annotation'][idx] = os.path.join(self.yurf_path,'annotations',annotation)
            for idx, ortho in enumerate(sites[key]['orthomosaic']):
                sites[key]['orthomosaic'][idx] = os.path.join(self.yurf_path,ortho)
        pprint.pprint(sites)
        return sites

    def collect_shonai_orthomosaics(self):
        sites = {"orthomosaic":[os.path.join(self.shonai_path,"orthomosiac_coastalforest.tif")],"annotation": os.path.join(self.shonai_path,"Annotations")}
        pprint.pprint(sites)
        return sites

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass