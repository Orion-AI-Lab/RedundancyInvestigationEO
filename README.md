# Repository for Redundancy Investigation
This repository contains code for the paper: Hide and Seek: Investigating Redundancy in Earth Observation Imagery (CV4EO, WACV 2027)

# Table of Contents
- [Setup Project](#setup-project)
- [Repository Structure](#repository-structure)
- [Downloading the data](#downloading-datasets)
- [Running Experiments](#experiments)
    - [What a reproduced run is](#what-a-reproduced-run-is)
    - [Before you train](#before-you-train)
    - [Webdataset setup](#webdataset-setup)
    - [Train one setting](#train-one-setting)
    - [Config files used for the main experiments](#config-files-used-for-the-main-experiments)
    -[Evaluate a saved checkpoint](#evaluate-a-saved-checkpoint)
- [Plotting data](#plotting-data)
- [Supported Models](#supported-models)
- [Supported datasets](#supported-datasets)
- [Extending the code](#extending-the-code)
    - [Adding new models](#adding-new-models)
    - [Add new augmentation methods](#adding-data-augmentations)
    - [Add new datasets](#adding-new-datasets)
    - [Adding new tasks](#adding-new-tasks)
-[Citation]


### Setup project

```
conda env create --name envname --python=3.10

pip install -r requirements_opt.txt, requirements_extra.txt (requirements.txt as a back-up option)
```

### Repository Structure
```css
.
├── configs/
│   ├── main_configs/$dataset_name$_configs.json
│   ├── datasets
│   ├── method
│   ├── download
│   └── training
│
├── datasets/
│   ├── BigEarthNetDataset.py
│   ├── FLAIRDataset.py
│   ├── MLRSNetDataset.py
│   ├── WaitituDataset.py
│   ├── WoodyDataset.py
│   └── MaridaDataset.py
│
├── downloading_scripts/
│    ├── bigearthnet.sh
│    ├── flair.sh
│    ├── mlrsnet.sh
│    ├── waititu.sh
│    ├── woody.sh
│    └── marida.sh
│
├── training/
│    └── classification.py
|    └── segmentation.py
│
└── utilities/
     └── augmentations.py
     └── utils.py
     └── model_utilities.py
     └── webdataset_writer.py

main.py
downloader.py
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



### Running Experiments

#### What a reproduced run is

One experiment is fully specified by four files plus the command-line overrides:

| Piece | Where it lives | What it controls |
| :--- | :--- | :--- |
| Run config | `configs/main_configs/<dataset>.json` | Dataset name, train or test, device, Weights & Biases, WebDataset paths, checkpoint root, seed fallback |
| Dataset config | `configs/datasets/<dataset>.json` | Task, classes, channels, image size, metrics. Loaded automatically from the dataset name |
| Method config | `configs/method/...` | Architecture, backbone, masking rule, retain ratio or similarity threshold, pretrained flag |
| Training config | `configs/training/training_per_dataset/...` | Epochs, batch size, optimizer, loss, augmentation, early stopping |

Checkpoints are written under:

```text
<checkpoint_root_path>/<task>/<dataset>/<architecture>/<backbone>/<wandb_run_name>/
```

With Weights & Biases off, the run name is `offline`. The best weights are `best_model_state_dict.pt`.

All information is aggregated by `main.py`. Given the aggregated configurations, the appropriate dataloading, training and testing functions are constructed.

 For each available task, the data loader follows same/similar patterns, enabling the training/testing procedures to remain (mostly) dataset agnostic.

 To run an experiment one has to select the desired dataset in `configs/main_configs/$dataset_name$.json`. The training options can be defined in `configs/training/training_per_dataset/$datastet_name$_training.json`.


If data augmentation is needed then set `augment:true` in `configs/training/training_per_dataset/$datastet_name$_training.json`. The desired data augmentations can be set in `configs/augmentations/augmentations.json`, along with their strength (probability of occuring) e.g if we want to always resize an image to 224x224 pixels set: 
```json5
"Resize":{
            "value":224,
            "p":1.0
        },
```
The current augmentation configuration files contains all supported augmentations.


#### Before you train

The JSON files currently contain absolute paths from the machine used to develop this repository (`/mnt/nvme1/tpapazaf/datasets`, `/home/tpapazaf/...`, and similar). Replace every path below with a directory on your machine. Configs accept `//` comments because they are parsed with `pyjson5`.

In `configs/main_configs/<dataset>.json`:

- `dataset_root_path` and `webdataset_root_path`: where the raw data and the WebDataset shards live.
- `checkpoint_root_path`: where weights are saved. `null` stores them in `checkpoints/` inside this repository.
- `device`: for example `cuda:0`.
- `phase`: `train` to fit a model, then evaluate. `test` only evaluates `eval_checkpoint`.
- `wandb`, `wandb_project`, `wandb_entity`: set `wandb` to `false` to train without an account. Run names then become `offline`.
- `dataset_percentage`: fraction of **training shards** to keep. `1.0` or omitting the key uses the full training set. 
- `seed` in the JSON is replaced by `--seed`.

In `configs/datasets/<dataset>.json`, set `root_path` to the same data root.

In the training JSON, set `class_weights_path` to a local file or to `null` so weights are computed from the loader. The FLAIR training file points at `configs/training/training_per_dataset/rvitupernet/flair_class_weights.pt` in this repository.



#### Webdataset setup

All datasets support the [webdataset](https://github.com/webdataset/webdataset) format for more efficient data loading. To enable it set `webdataset:true` in `configs/main_configs/$dataset_name$.json`. If webdataset shards exist, training begins immediately. Otherwise, the shards are created automatically. Depending on the dataset size this process may take from minutes to a few hours.

The example in `configs/configs.json` contains the following options for webdataset:
```
"webdataset":true,
"webdataset_shuffle_size": 1000,
"webdataset_initial_buffer":1000,
"max_samples_per_shard": 256, //set upper limit 256 samples per shard
"webdataset_root_path": null,
```
Setting the `webdtaset_root_path` variable will change the saving directory of the webdataset. If left at null, the webdataset will be saved at the same directory as the dataset of interest. The `max_samples_per_shard` argument is only used when creating the webdataset and refers to the maximum number of samples that would be contained in a single shard. This is handled by `utilities/webdataset_writer.py`. `webdataset_shuffle_size` determines the size of the buffer where the data are sampled, while `webdataset_initial_buffer` the amount of samples to be loaded before starting to yield. 




#### Train one setting

Classification (BigEarthNet, retain 60% of patches by farthest-point sampling on cosine similarity):

```bash
python main.py \
  --config configs/main_configs/bigearthnet.json \
  --method_config configs/method/classification/rvit.json \
  --training_config configs/training/training_per_dataset/rvit/bigearthnet_training.json \
  --seed 222 \
  --backbone vit_base_patch16_224 \
  --implementation "topx%" \
  --retain_percent 60 \
  --attention_mask false \
  --pretrained true
```

Segmentation (FLAIR, UPerNet decoder, same masking rule). Point `phase` at `train` and use the full training set before treating the numbers as a paper result:

```bash
python main.py \
  --config configs/main_configs/flair.json \
  --method_config configs/method/segmentation/rvitupernet.json \
  --training_config configs/training/training_per_dataset/rvitupernet/flair_training.json \
  --seed 222 \
  --backbone vit_base_patch16_224 \
  --implementation "topx%" \
  --retain_percent 75 \
  --attention_mask false \
  --pretrained true \
  --apply_mask_in_infer false
```

The flag name is `--method_config`. `run_gpu.sh` still passes `--method`. Align that script with `main.py` before you rely on the shell sweep.

After training, the test loader runs automatically. Metrics are printed and, when enabled, sent to Weights & Biases.

#### Config files used for the main experiments

| Dataset | Task | Main config | Method | Training |
| :--- | :--- | :--- | :--- | :--- |
| FLAIR | Segmentation | `configs/main_configs/flair.json` | `configs/method/segmentation/rvitupernet.json` | `configs/training/training_per_dataset/rvitupernet/flair_training.json` |
| Woody | Segmentation | `configs/main_configs/woody.json` | same method file | `.../rvitupernet/woody_training.json` |
| Waititu | Segmentation | `configs/main_configs/waititu.json` | same method file | `.../rvitupernet/waititu_training.json` |
| MARIDA | Segmentation | `configs/main_configs/marida.json` | same method file | `.../rvitupernet/marida_training.json` |
| BigEarthNet | Multi-label classification | `configs/main_configs/bigearthnet.json` | `configs/method/classification/rvit.json` | `configs/training/training_per_dataset/rvit/bigearthnet_training.json` |
| MLRSNet | Multi-label classification | `configs/main_configs/mlrsnet.json` | `configs/method/classification/rvit.json` | `configs/training/training_per_dataset/rvit/mlrsnet_training.json` |

`run_gpu.sh` is the sweep launcher. Edit the arrays at the top (`vit_backbones`, `implementations`, `retain_percents` or `mask_thresholds`), the three config paths, `seed`, `PRETRAINED`, and `APPLY_MASK_IN_INFER`, then:

```bash
./run_gpu.sh --gpu 0
```

That writes a temporary `configs/main_configs/<dataset>_gpu<N>.json` with `device` set to `cuda:<N>` and deletes it on exit. The commented threshold lists in `run_gpu.sh` are the operating points used while sweeping each dataset (similarity thresholds for `max_len`, retain percentages for `topx%`). Uncomment the list that matches the dataset you are reproducing and set `main_config`, `method_config`, and `training_config` to that dataset. 

#### Evaluate a saved checkpoint

In the main config set:

```json5
"phase": "test",
"eval_checkpoint": "/path/to/best_model_state_dict.pt"
```

Then call `main.py` with the **same** method and training configs used to train that checkpoint, including `--retain_percent`, `--implementation`, `--backbone`, and `--apply_mask_in_infer`. The checkpoint path in the committed configs is local and will not exist on another machine.

`apply_mask_in_infer` controls whether tokens are dropped at test time. `false` runs the full token sequence at test (the accuracy comparison). `true` applies the same retention rule at test (the efficiency measurement). `run_gpu.sh` sets this with `APPLY_MASK_IN_INFER`.



### Plotting data
Each dataset implements a plot() function to handle the visualization of the examples. For example, executing `dataset.plot(index=0)` for the BigEarthNet dataset returns 

![BigEarthNet example](examples/BigEarthNet_sample0.png)


### Supported models

For classification tasks we support all encoders available in [timm](https://github.com/huggingface/pytorch-image-models). 
In this benchmark we mainly focus on:

| Model | Paper | 
| :---: | :---: |
|ViT| [ViT Paper](https://arxiv.org/abs/2010.11929)|

In terms of semantic semgnetation problems we focus on:

| Model | Paper | 
| :---: | :---: |
|UperNet|[UperNet Paper](https://arxiv.org/abs/1807.10221)|

### Supported datasets

The following table presents the datasets supported in this repo, along with some basic information regarding the data sensor, their spatial coverage and the tasks they enable. Each dataset can be used in the respective config files with the following names in lower case.

| Dataset | Modalities | Possible Tasks | Covered Areas | 
| :---: | :---: | :---: | :---: |
| [FLAIR](https://arxiv.org/pdf/2211.12979.pdf) | Aerial - RGB, NIR, Elevation|Segmentation| France|
| [Woody](https://www.sciencedirect.com/science/article/abs/pii/S0034425719301166)| Aerial | Segmentation | Chile|
| [Waititu](https://zenodo.org/record/7648984#.Y_gBqLTMKBQ)| Aerial | Segmentation| New Zealand|
| [BigEarthNet](https://arxiv.org/pdf/2407.03653) | Sentinel-1, Sentinel-2| Multi-label Classification | Austria, Belgium, Finland, Ireland, Kosovo, Lithuania, Luxembourg, Portugal, Serbia, Switzerland|
| [MLRSNet](https://arxiv.org/abs/2010.00243) | MultiSensor | Multi-label Classification | Global (more than 100 countries)|
| [MARIDA](https://marine-debris.github.io/) | Sentinel-2 | Honduras, Guatemala, Haiti, Santo Domingo, Vietnam, South Africa, Scotland, Indonesia, Philippines, South Korea and China |

### Extending the code

**Model.** Build it in `create_classifier` or `create_segmentor` in `utilities/model_utilities.py`.

**Augmentation.** Set `augment` to `true` in the training config and add the transform in `get_augmentations` in `utilities/augmentations.py`. Probabilities live in `configs/augmentations/augmentations.json`.

**Dataset.** Add `configs/datasets/<name>.json`, a loader under `datasets/` that returns `(sample, label)` and implements `plot`, a branch in `load_dataset()` in `utilities/utils.py`, channel statistics in `configs/stats/stats.json` (`calc_stats.py` can compute them), and a download script registered in `downloader.py`.

#### Adding new models
To add support for a new model just include its construction in `utilities/model_utilities.py`. Depending on the task, the models are constructed in one of the following functions:
```
create_classifier() // for classification tasks, and
create_segmentor() // for semantic segmentation tasks
```

#### Adding data augmentations
The data augmentation pipeline is based on the [Albumentations](https://albumentations.ai/docs/) library. To add an augmentation method one should include it in the `get_augmentations` function of `utilities/augmentations.py`. For example to add the `VerticalFlip` augmentation we add:

```python
elif k == "VerticalFlip":
    aug = A.augmentations.VerticalFlip(p=v["p"])
```


#### Adding new datasets

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



### Citation

If you use our work, please cite:

```bibtex
@misc{Papazafeiropoulos2026redundancyinvestigationEO,
    title   = {Hide and Seek: Investigating Redundancy in Earth Observation Imagery},
    author  = {Tasos Papazafeiropoulos and Nikolaos Ioannis Bountos and Nikolas Papadopoulos and Ioannis Papoutsis},
    year={2026},
    eprint={2603.13524},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2603.13524},
}
```