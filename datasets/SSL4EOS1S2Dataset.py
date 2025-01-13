# Code is mostly based on https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/ssl4eo.py

import glob
import os
import random
from typing import Callable, Optional, TypedDict

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pyjson5 as json
import rasterio
import torch
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import (check_integrity, download_url,
                                     extract_archive)
from tqdm import tqdm
import utilities
import einops
from torchvision import transforms
import rioxarray as rio
import datetime
import pickle

class SSL4EOS1S2(NonGeoDataset):
    """SSL4EO-S12 dataset.

    `Sentinel-1/2 <https://github.com/zhu-xlab/SSL4EO-S12>`_ version of SSL4EO.

    The dataset consists of unlabeled patch triplets (Sentinel-1 dual-pol SAR,
    Sentinel-2 top-of-atmosphere multispectral, Sentinel-2 surface reflectance
    multispectral) from 251079 locations across the globe, each patch covering
    2640mx2640m and including four seasonal time stamps.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2211.07044

    .. note::

       This dataset can be downloaded using:

       .. code-block:: console

          $ export RSYNC_PASSWORD=m1660427.001
          $ rsync -av rsync://m1660427.001@dataserv.ub.tum.de/m1660427.001/ .

       The dataset is about 1.5 TB when compressed and 3.7 TB when uncompressed, and
       takes roughly 36 hrs to download, 1 hr to checksum, and 12 hrs to extract.

    .. versionadded:: 0.5
    """

    size = 264

    class _Metadata(TypedDict):
        filename: str
        md5: str
        bands: list[str]

    metadata: dict[str, _Metadata] = {
        "s1": {
            "filename": "s1.tar.gz",
            "md5": "51ee23b33eb0a2f920bda25225072f3a",
            "bands": ["VV", "VH"],
        },
        "s2c": {
            "filename": "s2_l1c.tar.gz",
            "md5": "b4f8b03c365e4a85780ded600b7497ab",
            "bands": [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B8A",
                "B9",
                "B10",
                "B11",
                "B12",
            ],
        },
        "s2a": {
            "filename": "s2_l2a.tar.gz",
            "md5": "85496cd9d6742aee03b6a1c99cee0ac1",
            "bands": [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B8A",
                "B9",
                "B11",
                "B12",
            ],
        },
    }

    def __init__(
        self,
        configs,
        mode='train',
        checksum: bool = False,
    ) -> None:
        """Initialize a new SSL4EOS12 instance.

        Args:
            root: root directory where dataset can be found
            split: one of "s1" (Sentinel-1 dual-pol SAR), "s2c" (Sentinel-2 Level-1C
                top-of-atmosphere reflectance), and "s2a" (Sentinel-2 Level-2a surface
                reflectance)
            seasons: number of seasonal patches to sample per location, 1--4
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if dataset is missing or checksum fails
        """
        split = configs['split']
        seasons = configs['seasons']
        for spl in split:
            assert spl in self.metadata
        
        assert seasons in range(1, 5)

        self.configs = configs
        root = os.path.join(configs['root_path'],'SSL4EOS1S2')
        self.root = root
        self.split = split
        self.seasons = seasons
        self.transforms = transforms
        self.checksum = checksum
        
        if not isinstance(split,list):
            self.subdir = os.path.join(self.root,split)
            self.split = [self.split]
        else:
            #select only one modality to list scenes. Scenes should be aligned.
            self.subdir = os.path.join(self.root,'s1')

        self.bands = {'s1':self.metadata['s1']["bands"],'s2a':self.metadata['s2a']['bands']}

        if self.configs['augment']:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None        
        if self.configs['normalization']=='standard':
            self.normalization = transforms.Normalize(mean=self.configs['mean'],std=self.configs['std'])

        self._verify()

        if mode!='train':
            self.scenes = []
            self.samples = []
        else:
            if not os.path.isfile(os.path.join(self.root,'samples.pkl')):
                self.scenes = sorted(os.listdir(self.subdir))
                self.samples = []
                for scene in tqdm(self.scenes):
                    patches_s1_dir = os.path.join(self.subdir,scene)
                    patches = os.listdir(patches_s1_dir)
                    s1_patches = []
                    s2_patches = []
                    for patch in patches:
                        sample_path_s1 = os.path.join(patches_s1_dir,patch)
                        s1_patches.append(sample_path_s1)
                    #s2 path
                    patches_s2_dir = patches_s1_dir.replace('s1','s2a')
                    patches = os.listdir(patches_s2_dir)
                    for patch in patches:
                        sample_path_s2= os.path.join(patches_s2_dir,patch)
                        s2_patches.append(sample_path_s2)
                    format_string = "%Y%m%d"
                    
                    tmp_s1 = list(zip(s1_patches, [datetime.datetime.strptime(x.split('/')[-1].split('_')[4][:8], format_string) for x in s1_patches]))
                    tmp_s2 = list(zip(s2_patches, [datetime.datetime.strptime(x.split('/')[-1].split('_')[0][:8],format_string) for x in s2_patches ]))
                    sorted_pairs_s1 = sorted(tmp_s1, key=lambda x: x[1])
                    sorted_pairs_s2 = sorted(tmp_s2, key=lambda x: x[1])
                    
                    for (s1,s2) in list(zip(sorted_pairs_s1,sorted_pairs_s2)):
                        self.samples.append({'s1':s1[0],'s2a':s2[0]})
                    
                random.Random(999).shuffle(self.samples)
                print('Saving samples pickle')
                with open(os.path.join(self.root,'samples.pkl'), 'wb') as f:
                    pickle.dump(self.samples, f)
            else:
                print('Loading samples pickle')
                with open(os.path.join(self.root,'samples.pkl'), 'rb') as f:
                    self.samples = pickle.load(f)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image sample
        """
        total_image = None
        meta = None
        for spl in self.split:
            
            directory = self.samples[index]
            
            directory = directory[spl]
            modality_image = []
            for band in self.bands[spl]:
                filename = os.path.join(directory, f"{band}.tif")
                #with rasterio.open(filename) as f:
                #    image = f.read(out_shape=(1, self.size, self.size))
                image = rio.open_rasterio(filename,engine='rasterio')
                image = image.rio.reproject(
                                image.rio.crs,
                                shape=(self.size, self.size),
                                resampling=rasterio.enums.Resampling.bilinear,
                            )
                lat = image.y.to_numpy()
                lon = image.x.to_numpy()
                image = image.to_numpy()
                
                modality_image.append(torch.from_numpy(image.astype(np.float32)))

                meta = {'lon':lon,'lat':lat}
                    
            #split_images.append(torch.from_numpy(image.astype(np.float32)))
            season_image = torch.cat(modality_image)
        
            if total_image is None:
                total_image = season_image
            else:
                total_image = torch.cat((total_image,season_image),dim=0)

        image = total_image.numpy()
        
        
        if not self.configs['webdataset']:
            if self.configs['augment'] and self.mode=="train":
                image = einops.rearrange(image, 'c h w -> h w c')
                transform = self.augmentations(image=image)
                image = transform['image']
                image = einops.rearrange(image, 'h w c -> c h w')
            if self.configs['normalization']=='minmax':
                image /= image.max()
            elif self.configs['normalization']=='standard':
                image = torch.from_numpy(image).float()
                image = self.normalization(image)
            else:
                image = torch.from_numpy(image).float()
        else:
            image = torch.from_numpy(image).float()
        label = {'label':0,'coordinates':{'lat':meta['lat'],'lon':meta['lon']}}
        return image, label


    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return 251079

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        directory_path = os.path.join(self.root, self.split[0])
        if os.path.exists(directory_path):
            return

        # Check if the zip files have already been downloaded
        filename = self.metadata[self.split]["filename"]
        zip_path = os.path.join(self.root, filename)
        
        md5 = self.metadata[self.split]["md5"] if self.checksum else None
        integrity = check_integrity(zip_path, md5)
        if integrity:
            print('Zip is valid')
            #self._extract()
        else:
            raise RuntimeError(f"Dataset not found in `root={self.root}`")

    def _extract(self) -> None:
        """Extract the dataset."""
        filename = self.metadata[self.split]["filename"]
        extract_archive(os.path.join(self.root, filename))

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        nrows = 2 if self.split == "s1" else 1
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=self.seasons,
            squeeze=False,
            figsize=(4 * self.seasons, 4 * nrows),
        )

        for i in range(self.seasons):
            image = sample["image"][i * len(self.bands) : (i + 1) * len(self.bands)]

            if self.split == "s1":
                axes[0, i].imshow(image[0])
                axes[1, i].imshow(image[1])
            else:
                image = image[[3, 2, 1]].permute(1, 2, 0)
                image = torch.clamp(image / 3000, min=0, max=1)
                axes[0, i].imshow(image)

            axes[0, i].axis("off")

            if show_titles:
                axes[0, i].set_title(f"Split {self.split}, Season {i + 1}")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

