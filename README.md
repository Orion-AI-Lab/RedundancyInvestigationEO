# Repository for Redundancy Investigation
This repository contains code for the investigation of redundancy on various datasets

# Table of Contents
- [Setup Project](#setup-project)
- [Repository Structure](#repository-structure)
- [Downloading the data](#downloading-datasets)
- [Running Experiments](#experiments)
    - [Supported Models](#supported-models)
    - [Adding new models](#adding-new-models)
    - [Adding new tasks](#adding-new-tasks)
    - [Training Foundation Models](#train-foundation-models)
    - [Investigate Redundancy](#investigate-redundancy)
- [Add new augmentation methods](#adding-data-augmentations)
- [Add new datasets](#adding-new-datasets)
- [Webdataset setup](#webdataset-setup)
- [Plotting data](#plotting-data)
- [Supported datasets](#datasets-in-the-benchmark)

### Setup project

```
conda env create --name envname --python=3.10

pip install -r requirements_opt.txt, requirements_extra.txt (requirements.txt as a back-up option)
```

### Repository Structure
```css
.
├── configs/
│   ├── configs.json
│   ├── datasets
│   ├── method
│   ├── download
│   └── training
├── datasets/
│   ├── BigEarthNetDataset.py
│   ├── FLAIRDataset.py
│   └──  ...
├── downloading_scripts/
|    ├── bigearthnet.sh
|    ├── flair.sh
|    └── ...
├── training/
|    └── classification.py
|    └── segmentation.py
└── utilities/
     └── augmentations.py
     └── utils.py
     └── model_utilities.py
     └── webdataset_writer.py
main.py
downloader.py
calc_stats.py
```

`configs.json` contains high-level experimental choices, including the dataset of interest and whether to activate wandb.


the `datasets/` directory contains the dataset specific configurations e.g task to solve, metrics to log etc. 

Example configurations for the `bigearthnet` dataset:

```json5
{
    "root_path": "/mnt/nvme/tpapazaf/datasets",
    "task": "classification", // Possible Tasks: Depending on the dataset,
    "metrics": [
        "accuracy",
        "fscore",
        "map"
    ], //Desired metrics to log
    "num_classes": 19,
    "in_channels": 14,
    "multilabel": true,
    "image_size": 120,
    "meta_info": "Data collected in Europe.",
    "class_names": ["Urban fabric", "Industrial or commercial units", "Arable land", "Permanent crops", "Pastures", "Complex cultivation patterns", "Land principally occupied by agriculture, with significant areas of natural vegetation", "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest", "Mixed forest", "Natural grassland and sparsely vegetated areas", "Moors, heathland and sclerophyllous vegetation", "Transitional woodland-shrub", "Beaches, dunes, sands", "Inland wetlands", "Coastal wetlands", "Inland waters", "Marine waters"]
}
```

Similarly, the method and training directories contain the configuration choices for the desired tasks e.g classification, segmentation and training choices e.g batch size, epochs etc respectively.


### Downloading Datasets

To download the datasets used in this benchmark select the desired dataset in `configs/download/download.json` along with the base directory to store the data and execute

 `python downloader.py`. 
 
 Make sure to give proper permissions to the scripts under `downloading_scripts/` running:
 
  `chmod +x download_script.sh`

`downloader.py` will handle the downloading and all necessary restructuring needed for the experiments.   


### Experiments

All information is aggregated by `main.py`. Given the aggregated configurations, the appropriate dataloading, training and testing functions are constructed.

 For each available task, the data loader follows same/similar patterns, enabling the training/testing procedures to remain (mostly) dataset agnostic.

 To run an experiment one has to select the desired dataset in `configs/main_configs/$dataset_name$.json`. The training options can be defined in `configs/training/training_per_dataset/$datastet_name$_training.json`.

 All datasets support the [webdataset](https://github.com/webdataset/webdataset) format for more efficient data loading. To enable it set `webdataset:true` in `configs/main_configs/$dataset_name$.json`. If webdataset shards exist, training begins immediately. Otherwise, the shards are created automatically. Depending on the dataset size this process may take from minutes to a few hours.

If data augmentation is needed then set `augment:true` in `configs/training/training_per_dataset/$datastet_name$_training.json`. The desired data augmentations can be set in `configs/augmentations/augmentations.json`, along with their strength (probability of occuring) e.g if we want to always resize an image to 224x224 pixels set: 
```json5
"Resize":{
            "value":224,
            "p":1.0
        },
```
The current augmentation configuration files contains all supported augmentations.

#### Webdataset setup
The example in `configs/configs.json` contains the following options for webdataset:
```
"webdataset":true,
"webdataset_shuffle_size": 1000,
"webdataset_initial_buffer":1000,
"max_samples_per_shard": 256, //set upper limit 256 samples per shard
"webdataset_root_path": null,
```
Setting the `webdtaset_root_path` variable will change the saving directory of the webdataset. If left at null, the webdataset will be saved at the same directory as the dataset of interest. The `max_samples_per_shard` argument is only used when creating the webdataset and refers to the maximum number of samples that would be contained in a single shard. This is handled by `utilities/webdataset_writer.py`. `webdataset_shuffle_size` determines the size of the buffer where the data are sampled, while `webdataset_initial_buffer` the amount of samples to be loaded before starting to yield. 
#### Supported models

For classification tasks we support all encoders available in [timm](https://github.com/huggingface/pytorch-image-models). 
In this benchmark we mainly focus on:

| Model | Paper | 
| :---: | :---: |
|ViT| [ViT Paper](https://arxiv.org/abs/2010.11929)|

In terms of semantic semgnetation problems we focus on:

| Model | Paper | 
| :---: | :---: |
|UperNet|[UperNet Paper](https://arxiv.org/abs/1807.10221)|

#### Adding new models
To add support for a new model just include its construction in `utilities/model_utilities.py`. Depending on the task, the models are constructed in one of the following functions:
```
create_classifier() // for classification tasks, and
create_segmentor() // for semantic segmentation tasks
```


#### Investigate Redundancy

Example for cosine similarity threshold set to 60\%:
```
python main.py --config=configs/main_configs/bigearthnet.json --seed=222 --method=configs/method/classification/rvit.json --training_config=configs/training/training_per_dataset/rvit/bigearthnet_training.json --mask_threshold=0.6
```

### Adding data augmentations
The data augmentation pipeline is based on the [Albumentations](https://albumentations.ai/docs/) library. To add an augmentation method one should include it in the `get_augmentations` function of `utilities/augmentations.py`. For example to add the `VerticalFlip` augmentation we add:

```python
elif k == "VerticalFlip":
    aug = A.augmentations.VerticalFlip(p=v["p"])
```


 ### Adding new datasets

 To add a new dataset one has to:
    
    1. Create a configuration file in configs/datasets/ with the name of the dataset in lower case (e.g configs/datasets/flair.json)
        - The configuration file has to include:
            - The root path of the data
            - The nature of the task to solve (e.g classification)
            - The metrics to log in a list (e.g ["accuracy","fscore"])
            - Any other information needed for loading the data (depends on your implementation of the data loader).
    2. Create a data loader in datasets/ (e.g datasets/FLAIRDataset.py). Each dataset should include a plot() function for visualization.
    3. The data loader should return data in the following form: `sample, label`. If a different scheme is used, the training procedures should be adapted accordingly (e.g classification.py and the webdataset processing pipeline.)
    4. Include the option to load the dataset in the load_dataset() function of the utilities/utils.py
        - The following code block shows an example for the FLAIR dataset
    5. Include the mean and std of the new dataset to configs/stats/stats.json. These stats can be calculated using the calc_stats.py script. Set the option batched=True to process the dataset in batches.
```python
elif configs['dataset'].lower()=='flair':
    dataset = datasets.FLAIRDataset.FLAIRDataset(configs,mode)
```

### Plotting data
Each dataset implements a plot() function to handle the visualization of the examples. For example, executing `dataset.plot(index=0)` for the BigEarthNet dataset returns 

![BigEarthNet example](examples/BigEarthNet_sample0.png)



### Datasets in the benchmark
The following table presents the datasets supported in this repo, along with some basic information regarding the data sensor, their spatial coverage and the tasks they enable. Each dataset can be used in the respective config files with the following names in lower case.


| Dataset | Modalities | Possible Tasks | Covered Areas | 
| :---: | :---: | :---: | :---: |
| [FLAIR](https://arxiv.org/pdf/2211.12979.pdf) | Aerial - RGB, NIR, Elevation|Segmentation| France|
| [FLAIR2](https://arxiv.org/pdf/2305.14467.pdf) | Aerial - RGB, NIR, Elevation, Sentinel-2| France|
| [Woody](https://www.sciencedirect.com/science/article/abs/pii/S0034425719301166)| Aerial | Segmentation | Chile|
| [Waititu](https://zenodo.org/record/7648984#.Y_gBqLTMKBQ)| Aerial | Segmentation| New Zealand|
| [BigEarthNet-MM](https://arxiv.org/pdf/2105.07921.pdf) | Sentinel-1, Sentinel-2| Multi-label Classification | Austria, Belgium, Finland, Ireland, Kosovo, Lithuania, Luxembourg, Portugal, Serbia, Switzerland|


