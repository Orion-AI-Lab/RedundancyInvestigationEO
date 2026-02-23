#!/usr/bin/env python3
"""
Visualize masking at inference for classification and segmentation.

Loads a model from checkpoint, runs a single sample with masking, and saves a figure:
- Classification: raw image, masked image, ground truth label, model prediction.
- Segmentation: raw image, masked image, ground truth mask, model prediction (side by side).

Sample can be loaded from a file path (--sample_source path) or from webdataset (--sample_source webdataset).

Usage:
  # From webdataset (FLAIR): split + sample index
  python visualize_masking.py --task segmentation \
    --checkpoint_path /path/to/ckpt --config configs/main_configs/flair.json \
    --method_config configs/method/segmentation/rvitupernet.json \
    --training_config configs/training/training_per_dataset/rvitupernet/flair_training.json \
    --webdataset_split test --sample_index 0 --output_path out_seg.png

  # From file path (FLAIR): infer label from image path
  python visualize_masking.py --task segmentation --sample_source path \
    ... --sample_path /path/to/flair_aerial_train/area/sub/img/XXX_IMG.tif --output_path out_seg.png

  # Classification from webdataset
  python visualize_masking.py --task classification --sample_source webdataset \
    ... --webdataset_split val --sample_index 5 --output_path out_cls.png
"""

import argparse
import glob
import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds

# PIL fallback when matplotlib hits numpy compat issues (e.g. ERR_IGNORE)
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# rasterio only needed when sample_source=path (loading .tif from disk)

# Project imports
import utilities.utils as utils
import utilities.model_utilities as model_utils

# Built-in class names for classification visualization (index -> display name).
# Override by adding "class_names" to the dataset config (configs/datasets/<dataset>.json).
# Mapping: index i must match the label index used by the dataset and model. To verify:
#   python utilities/verify_classification_mapping.py bigearthnet [--config configs/main_configs/your_bigearthnet.json]
#   python utilities/verify_classification_mapping.py mlrsnet
# Then ensure configs/datasets/<dataset>.json "class_names" is in the same order as the script prints.
BIGEARTHNET_19_CLASS_NAMES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland-shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]
# MLRSNet: 60 multi-labels; exact names are in Categories_names.xlsx. Use placeholders or add "class_names" in configs/datasets/mlrsnet.json.
MLRSNET_60_CLASS_NAMES = [f"class_{i}" for i in range(60)]

DATASET_CLASS_NAMES = {
    "bigearthnet": BIGEARTHNET_19_CLASS_NAMES,
    "mlrsnet": MLRSNET_60_CLASS_NAMES,
}


def _load_mlrsnet_class_names_from_dataset(root_path):
    """Load MLRSNet label names from labels/*.csv column headers (index order = column order). Returns list of 60 or None."""
    if not root_path:
        return None
    root_path = os.path.expandvars(os.path.abspath(root_path))
    labels_dir = os.path.join(root_path, "MLRSNet", "labels")
    if not os.path.isdir(labels_dir):
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    csv_files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".csv"))
    if not csv_files:
        return None
    path = os.path.join(labels_dir, csv_files[0])
    try:
        df = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    cols = df.columns.tolist()
    # First column is "image"; next 60 are label names (same order as MLRSNetDataset label_keys)
    if len(cols) < 2:
        return None
    names = [str(c).strip() for c in cols[1:61]]  # up to 60 names
    names = [n for n in names if n]
    if len(names) != 60:
        return None
    return names[:60]


def _load_bigearthnet_class_names_from_dataset(configs):
    """Load BigEarthNet-19 label names in the exact index order used by the dataset (labels_keys). Returns list of 19 or None."""
    try:
        from datasets.BigEarthNetDataset import BigEarthNetDataset
    except ImportError:
        return None
    try:
        ds = BigEarthNetDataset(configs, mode="train")
    except Exception:
        return None
    # labels_keys: name -> index; build index -> name
    names_by_idx = [None] * len(ds.labels_keys)
    for name, idx in ds.labels_keys.items():
        names_by_idx[idx] = name
    if all(n is not None for n in names_by_idx):
        return names_by_idx
    return None


def load_configs_for_viz(args):
    """Build config dict from args (same structure as main.py load_configs)."""
    from types import SimpleNamespace
    ns = SimpleNamespace(
        config=args.config,
        training_config=args.training_config,
        dataset_config=None,
        method_config=args.method_config,
        augmentation_config=None,
        wandb_id_resume=None,
        seed=args.seed,
        mask_threshold=None,
        implementation=None,
        attention_mask=None,
        backbone=getattr(args, "backbone", None),
        retain_percent=None,
        pretrained=None,
        apply_mask_in_infer=args.apply_mask_in_infer,
    )
    configs = utils.load_configs(ns)
    # Prefer backbone from checkpoint path: .../rvitupernet/vit_base_patch16_224/sleek-cloud-7/best_model_state_dict.pt
    ckpt_path = getattr(args, "checkpoint_path", None)
    if ckpt_path:
        # path like .../rvitupernet/vit_base_patch16_224/sleek-cloud-7/best_model_state_dict.pt
        ckpt_dir = os.path.dirname(ckpt_path)
        parent = os.path.dirname(ckpt_dir)
        inferred = os.path.basename(parent)
        skip = ("segmentation", "classification", "flair", "rvitupernet", "Checkpoints", "FlairCheckpoints")
        if inferred and inferred not in skip and not inferred.startswith("."):
            configs["backbone"] = inferred
            print(f"Using backbone from checkpoint path: {configs['backbone']}")
    if not configs.get("backbone"):
        backbone_arg = getattr(args, "backbone", None)
        if backbone_arg:
            configs["backbone"] = backbone_arg
        else:
            raise ValueError("Could not infer backbone from checkpoint path. Pass --backbone (e.g. vit_base_patch16_224).")
    # For classification: load real label names from dataset when possible (MLRSNet from CSV, BigEarthNet from dataset)
    dataset_key = (configs.get("dataset") or "").lower()
    if configs.get("task") == "classification":
        if dataset_key == "mlrsnet":
            root = os.path.expandvars(configs.get("root_path") or "")
            names = _load_mlrsnet_class_names_from_dataset(root)
            if names and len(names) == configs.get("num_classes", 60):
                configs["class_names"] = names
                print(f"Loaded {len(names)} MLRSNet label names from dataset (labels CSV).")
            elif not configs.get("class_names"):
                configs["class_names"] = DATASET_CLASS_NAMES.get("mlrsnet", MLRSNET_60_CLASS_NAMES)
        elif dataset_key == "bigearthnet":
            names = _load_bigearthnet_class_names_from_dataset(configs)
            if names and len(names) == configs.get("num_classes", 19):
                configs["class_names"] = names
                print(f"Loaded {len(names)} BigEarthNet label names from dataset (exact index order).")
            elif not configs.get("class_names") and dataset_key in DATASET_CLASS_NAMES:
                configs["class_names"] = DATASET_CLASS_NAMES[dataset_key]
        elif not configs.get("class_names") and dataset_key in DATASET_CLASS_NAMES:
            configs["class_names"] = DATASET_CLASS_NAMES[dataset_key]
    return configs


def infer_label_path_from_flair(sample_path, root_path, labels_dir_name="flair_labels_train"):
    """Infer FLAIR label path from image path. Image is under flair_aerial_train/.../img/; label under flair_labels_train/.../msk/ with IMG->MSK."""
    # e.g. .../flair_aerial_train/area/sub/img/XXX_IMG.tif -> .../flair_labels_train/area/sub/msk/XXX_MSK.tif
    parts = sample_path.replace("\\", "/").split("/")
    if "flair_aerial_train" in parts:
        idx = parts.index("flair_aerial_train")
    elif "flair_1_aerial_test" in parts:
        idx = parts.index("flair_1_aerial_test")
        labels_dir_name = "flair_1_labels_test"
    else:
        return None
    base = parts[:idx]
    rest = parts[idx + 1:]  # area, sub_area, img, filename
    if len(rest) < 3:
        return None
    # rest = [area, sub_area, 'img', filename]
    area, sub_area, _, filename = rest[0], rest[1], rest[2], rest[3]
    label_filename = filename.replace("IMG", "MSK").replace("img", "msk")
    label_path = os.path.join("/".join(base), labels_dir_name, area, sub_area, "msk", label_filename)
    return label_path


def load_sample_from_path(sample_path, label_path, configs, device):
    """Load image and optional label from paths; apply dataset normalization; return (image_tensor, label_tensor or None)."""
    import rasterio
    with rasterio.open(sample_path) as src:
        image = src.read()  # (C, H, W)
    image = torch.from_numpy(image).float()
    if image.dim() == 2:
        image = image.unsqueeze(0)
    # Normalize
    if configs.get("normalization") == "standard" and "mean" in configs and "std" in configs:
        mean = torch.tensor(configs["mean"], dtype=image.dtype).view(-1, 1, 1)
        std = torch.tensor(configs["std"], dtype=image.dtype).view(-1, 1, 1)
        image = (image - mean) / std
    elif configs.get("normalization") == "minmax":
        image = image / (image.max() + 1e-8)
    # Resize to config image_size if needed
    target_size = configs.get("image_size", image.shape[-1])
    if image.shape[-1] != target_size or image.shape[-2] != target_size:
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=(target_size, target_size), mode="bilinear", align_corners=False)
        image = image.squeeze(0)
    image = image.unsqueeze(0).to(device)  # (1, C, H, W)
    label = None
    if label_path and os.path.isfile(label_path):
        with rasterio.open(label_path) as src:
            label = src.read()
        label = torch.from_numpy(label).long().squeeze()
        if label.dim() == 0:
            label = label.unsqueeze(0)
        # FLAIR: raster 1..19 -> 0..18
        label = label - 1
        if image.shape[-1] != label.shape[-1] or image.shape[-2] != label.shape[-2]:
            label = label.unsqueeze(0).unsqueeze(0).float()
            label = F.interpolate(label, size=(image.shape[2], image.shape[3]), mode="nearest")
            label = label.squeeze().long()
        label = label.unsqueeze(0).to(device)  # (1, H, W)
    return image, label


def load_sample_from_webdataset(configs, split, sample_index, device):
    """
    Load a single sample from webdataset by split (train/val/test) and sample index.
    Applies the same normalization/resize as the eval pipeline. Returns (image, label).
    image: (1, C, H, W) on device; label: (1, H, W) for segmentation or (1,) for classification.
    """
    if "webdataset_root_path" not in configs or configs["webdataset_root_path"] is None:
        webdataset_path = os.path.join(configs.get("root_path", ""), "webdataset", configs["dataset"])
    else:
        webdataset_path = os.path.join(
            os.path.expandvars(configs["webdataset_root_path"]),
            "webdataset",
            configs["dataset"],
        )
    shard_dir = os.path.join(webdataset_path, split)
    if not os.path.isdir(shard_dir):
        raise FileNotFoundError(f"Webdataset split dir not found: {shard_dir}")
    shards = sorted(glob.glob(os.path.join(shard_dir, f"sample-{split}-*.tar")))
    if not shards:
        raise FileNotFoundError(f"No shards found in {shard_dir}")

    def process_sample(sample):
        image = torch.load(io.BytesIO(sample["image.pth"])).float()
        label = torch.load(io.BytesIO(sample["labels.pth"]))
        if isinstance(label, dict):
            label = label.get("label", label)
        if configs.get("dataset") == "marida":
            import einops
            image = einops.rearrange(image, "h w c -> c h w")
            label[label == -1] = 11
        if configs.get("dataset") == "woody":
            image = image[:3, :, :]
        if configs.get("enforce_resize") is not None:
            import albumentations as A
            target = configs["enforce_resize"]
            if configs["task"] == "segmentation":
                image = image.permute(1, 2, 0).numpy()
                aug = A.Compose([A.augmentations.Resize(height=target, width=target, p=1.0)])(image=image, mask=label.numpy())
                image = torch.from_numpy(aug["image"]).float().permute(2, 0, 1)
                label = torch.from_numpy(aug["mask"]).long()
            else:
                image = image.permute(1, 2, 0).numpy()
                aug = A.Compose([A.augmentations.Resize(height=target, width=target, p=1.0)])(image=image)
                image = torch.from_numpy(aug["image"]).float().permute(2, 0, 1)
        if configs.get("normalization") == "minmax":
            image = image / (image.max() + 1e-6)
        elif configs.get("normalization") == "standard" and "mean" in configs and "std" in configs:
            mean = torch.tensor(configs["mean"], dtype=image.dtype).view(-1, 1, 1)
            std = torch.tensor(configs["std"], dtype=image.dtype).view(-1, 1, 1)
            image = (image - mean) / std
        # Webdataset stores labels as returned by dataset (FLAIR already 0..18)
        label = label.long()
        if label.dim() == 0:
            label = label.unsqueeze(0)
        return image, label

    dataset = wds.WebDataset(shards, shardshuffle=False)
    count = 0
    for sample in dataset:
        image, label = process_sample(sample)
        if count == sample_index:
            target_size = configs.get("image_size", image.shape[-1])
            if image.shape[-1] != target_size or image.shape[-2] != target_size:
                image = image.unsqueeze(0)
                image = F.interpolate(image, size=(target_size, target_size), mode="bilinear", align_corners=False)
                image = image.squeeze(0)
            if label.dim() >= 2 and (label.shape[-1] != target_size or label.shape[-2] != target_size):
                label = label.unsqueeze(0).unsqueeze(0).float()
                label = F.interpolate(label, size=(target_size, target_size), mode="nearest")
                label = label.squeeze().long()
            image = image.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            return image, label
        count += 1
    raise IndexError(f"sample_index {sample_index} out of range (found {count} samples in {split})")


def build_masked_image(image_tensor, mask_bool, fill_value=0.0):
    """image_tensor (1, C, H, W), mask_bool (1, H, W). Return (1, C, H, W) with non-kept pixels set to fill_value."""
    mask = mask_bool.unsqueeze(1).float().to(image_tensor.device)
    if mask.shape[-2:] != image_tensor.shape[-2:]:
        mask = F.interpolate(mask, size=image_tensor.shape[-2:], mode="nearest")
    return image_tensor * mask + fill_value * (1 - mask)


def rgb_from_tensor(img_tensor, channel_indices=(0, 1, 2)):
    """(1, C, H, W) -> (H, W, 3) uint8 for display. channel_indices for RGB."""
    x = img_tensor[0].detach().cpu().float()
    # Ensure dense (avoid sparse_coo permute error)
    if x.is_sparse:
        x = x.to_dense()
    # Collapse to 3D (C, H, W); e.g. (1, C, H, W) from classification path
    while x.dim() > 3:
        x = x[0]
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() < 2 or x.numel() == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if x.shape[0] < 3 and x.shape[0] > 0:  # Handle single-channel images, repeat to 3 for display
        x = x.repeat(3, 1, 1)
    if x.shape[0] == 0:  # Handle empty images
        h, w = max(1, x.shape[-2] if x.dim() >= 2 else 1), max(1, x.shape[-1] if x.dim() >= 2 else 1)
        return np.zeros((h, w, 3), dtype=np.uint8)
    # Select RGB channels by channel_indices [R, G, B], or first 3 if indices not valid
    nch = x.shape[0]
    if (len(channel_indices) == 3 and
            all(0 <= i < nch for i in channel_indices)):
        x = x[list(channel_indices)]
    else:
        x = x[:min(3, nch)]
    if x.shape[0] < 3:
        x = x.repeat(3, 1, 1)
    if x.dim() != 3:
        return np.zeros((max(1, x.shape[-2]), max(1, x.shape[-1]), 3), dtype=np.uint8)
    x = x.permute(1, 2, 0).numpy()
    # Normalize for display: support [0,255], [0,1], or standardized (e.g. FLAIR) data
    x_min, x_max = float(x.min()), float(x.max())
    if x_max - x_min > 1e-6:
        x = (x - x_min) / (x_max - x_min) * 255.0
    else:
        x = np.zeros_like(x)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    return x


# Tab20-like RGB (0-255) for mask visualization without matplotlib colormap
_TAB20_RGB = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
    (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213),
    (196, 156, 148), (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]


def _mask_to_rgb(mask_2d, num_classes, color_list=None):
    """(H, W) int class indices -> (H, W, 3) uint8 RGB using color list."""
    color_list = color_list or _TAB20_RGB
    n = max(num_classes, len(color_list))
    colors = np.array([color_list[i % len(color_list)] for i in range(n)], dtype=np.uint8)
    h, w = mask_2d.shape
    idx = np.clip(mask_2d.ravel(), 0, n - 1)
    out = colors[idx].reshape(h, w, 3)
    return out


def _save_figure_pil(panels, titles, output_path, dpi=150):
    """Save 4 panels (list of (H,W,3) uint8) with titles using PIL. Avoids matplotlib draw/savefig."""
    if Image is None:
        raise RuntimeError("PIL is required for fallback save when matplotlib fails")
    h, w = panels[0].shape[0], panels[0].shape[1]
    title_h = 28
    panel_w = w
    total_w = 4 * panel_w
    total_h = title_h + h
    out = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    for i, (panel, title) in enumerate(zip(panels, titles)):
        x0 = i * panel_w
        out[title_h : title_h + h, x0 : x0 + panel_w] = panel
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for i, title in enumerate(titles):
        x0 = i * panel_w + panel_w // 2
        draw.text((x0, title_h // 2 - 7), title, fill=(0, 0, 0), anchor="mm", font=font)
    scale = dpi / 100.0
    new_w, new_h = int(total_w * scale), int(total_h * scale)
    if new_w != total_w or new_h != total_h:
        resample = getattr(getattr(Image, "Resampling", type(None)), "LANCZOS", None) or getattr(Image, "LANCZOS", 1)
        img = img.resize((new_w, new_h), resample)
    img.save(output_path, dpi=(dpi, dpi))


def _save_figure_pil_column(panels, titles, output_path, dpi=150):
    """Save 4 panels stacked vertically (column layout). Panels can have different sizes; use common width."""
    if Image is None:
        raise RuntimeError("PIL is required for fallback save when matplotlib fails")
    max_w = max(p.shape[1] for p in panels)
    title_h = 28
    out_rows = []
    for panel, title in zip(panels, titles):
        ph, pw = panel.shape[0], panel.shape[1]
        row_h = title_h + ph
        row = np.ones((row_h, max_w, 3), dtype=np.uint8) * 255
        x0 = (max_w - pw) // 2
        row[title_h : title_h + ph, x0 : x0 + pw] = panel
        out_rows.append(row)
    total_h = sum(r.shape[0] for r in out_rows)
    out = np.ones((total_h, max_w, 3), dtype=np.uint8) * 255
    y = 0
    for row in out_rows:
        out[y : y + row.shape[0], :] = row
        y += row.shape[0]
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    y = 0
    for i, title in enumerate(titles):
        draw.text((max_w // 2, y + title_h // 2 - 7), title, fill=(0, 0, 0), anchor="mm", font=font)
        y += panels[i].shape[0] + title_h
    scale = dpi / 100.0
    new_w, new_h = int(max_w * scale), int(total_h * scale)
    if new_w != max_w or new_h != total_h:
        resample = getattr(getattr(Image, "Resampling", type(None)), "LANCZOS", None) or getattr(Image, "LANCZOS", 1)
        img = img.resize((new_w, new_h), resample)
    img.save(output_path, dpi=(dpi, dpi))


def segmentation_visualization(raw_img, masked_img, gt_label, pred_label, mask_bool, num_classes, output_path):
    """Four panels: raw, masked, ground truth, prediction."""
    rgb = rgb_from_tensor(raw_img)
    rgb_m = rgb_from_tensor(masked_img)
    # Force masked-out (dropped) patches to black in the masked panel
    mask_np = mask_bool[0].detach().cpu().numpy()
    h, w = rgb_m.shape[0], rgb_m.shape[1]
    if mask_np.shape[0] != h or mask_np.shape[1] != w:
        if Image is not None:
            mask_pil = Image.fromarray(mask_np.astype(np.uint8))
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
            mask_np = np.array(mask_pil).astype(bool)
        else:
            # Nearest-neighbor resize without PIL
            yi = np.linspace(0, mask_np.shape[0] - 1, h).astype(np.intp)
            xi = np.linspace(0, mask_np.shape[1] - 1, w).astype(np.intp)
            mask_np = mask_np[np.ix_(yi, xi)].astype(bool)
    else:
        mask_np = mask_np.astype(bool)
    rgb_m[~mask_np] = 0
    gt = gt_label[0].detach().cpu().numpy() if gt_label is not None else np.zeros_like(pred_label[0].cpu().numpy())
    gt = np.clip(gt, 0, num_classes - 1).astype(np.int32)
    pred = pred_label[0].detach().cpu().numpy()
    pred = np.clip(pred, 0, num_classes - 1).astype(np.int32)
    gt_rgb = _mask_to_rgb(gt, num_classes)
    pred_rgb = _mask_to_rgb(pred, num_classes)

    try:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        cmap = plt.get_cmap("tab20", max(num_classes, 20))
        axes[0].imshow(rgb)
        axes[0].set_title("Raw image")
        axes[0].axis("off")
        axes[1].imshow(rgb_m)
        axes[1].set_title("Masked image")
        axes[1].axis("off")
        axes[2].imshow(gt_rgb)
        axes[2].set_title("Ground truth")
        axes[2].axis("off")
        axes[3].imshow(pred_rgb)
        axes[3].set_title("Model prediction")
        axes[3].axis("off")
        try:
            fig.tight_layout()
        except Exception:
            fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.05)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        panels = [rgb, rgb_m, gt_rgb, pred_rgb]
        titles = ["Raw image", "Masked image", "Ground truth", "Model prediction"]
        _save_figure_pil(panels, titles, output_path, dpi=150)
    print(f"Saved segmentation visualization to {output_path}")


def _wrap_line(line, max_chars=50):
    """Split a long line into multiple lines (by comma+space or by length)."""
    if len(line) <= max_chars:
        return [line]
    out = []
    for part in line.split(", "):
        part = part.strip()
        if not part:
            continue
        if out and len(out[-1]) + len(part) + 2 <= max_chars:
            out[-1] += ", " + part
        else:
            out.append(part)
    return out if out else [line[:max_chars]]


def _text_panel_image(h, w, text_lines, bg_rgb=(255, 255, 255), max_chars=50):
    """Draw text on a (h, w) RGB image; return (H, W, 3) uint8. Long lines are wrapped."""
    if Image is None:
        return np.ones((h, w, 3), dtype=np.uint8) * 255
    img = Image.new("RGB", (w, h), color=bg_rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    wrapped = []
    for line in text_lines:
        wrapped.extend(_wrap_line(str(line), max_chars=max_chars))
    line_height = 22
    y = max(12, h // 2 - (len(wrapped) * line_height) // 2)
    for line in wrapped:
        draw.text((w // 2, y), line, fill=(0, 0, 0), anchor="mm", font=font)
        y += line_height
    return np.array(img)


def _text_panel_image_sized(text_lines, panel_width=1000, line_height=22, max_chars=70):
    """Create a text panel image with enough width/height to show all text (no truncation)."""
    wrapped = []
    for line in text_lines:
        wrapped.extend(_wrap_line(str(line), max_chars=max_chars))
    h = max(120, len(wrapped) * line_height + 40)
    w = panel_width
    return _text_panel_image(h, w, text_lines, bg_rgb=(255, 255, 255), max_chars=max_chars)


def classification_visualization(raw_img, masked_img, gt_classes, pred_classes, class_names, output_path, mask_bool=None):
    """Four panels: raw, masked, and two text panels for GT and prediction.
    gt_classes / pred_classes: list of class indices (one for single-label, multiple for multilabel).
    If mask_bool is given, masked-out pixels are forced to black."""
    rgb = rgb_from_tensor(raw_img)
    rgb_m = rgb_from_tensor(masked_img)
    if mask_bool is not None:
        mask_np = mask_bool[0].detach().cpu().numpy()
        h, w = rgb_m.shape[0], rgb_m.shape[1]
        if mask_np.shape[0] != h or mask_np.shape[1] != w:
            if Image is not None:
                mask_pil = Image.fromarray(mask_np.astype(np.uint8))
                mask_pil = mask_pil.resize((w, h), Image.NEAREST)
                mask_np = np.array(mask_pil).astype(bool)
            else:
                yi = np.linspace(0, mask_np.shape[0] - 1, h).astype(np.intp)
                xi = np.linspace(0, mask_np.shape[1] - 1, w).astype(np.intp)
                mask_np = mask_np[np.ix_(yi, xi)].astype(bool)
        else:
            mask_np = mask_np.astype(bool)
        rgb_m[~mask_np] = 0
    # Format class indices as names (supports single-label and multilabel)
    def fmt_classes(indices):
        if not indices:
            return "(none)"
        if class_names and len(class_names) > 0:
            return ", ".join(class_names[i] if 0 <= i < len(class_names) else str(i) for i in indices)
        return ", ".join(str(i) for i in indices)
    gt_str = fmt_classes(gt_classes)
    pred_str = fmt_classes(pred_classes)
    # For multilabel, "Match" = exact subset match; test accuracy is often F1/micro-avg over labels, so per-sample match can differ
    match_str = "Match: yes" if set(gt_classes) == set(pred_classes) else "Match: no"
    h_img, w_img = rgb.shape[0], rgb.shape[1]
    # Text panels with enough width so "Ground truth:", "Prediction:" and full label lists are visible
    text_width = max(900, w_img * 3)
    gt_panel = _text_panel_image_sized([f"Ground truth:", gt_str], panel_width=text_width, max_chars=70)
    pred_panel = _text_panel_image_sized([f"Prediction:", pred_str, match_str], panel_width=text_width, max_chars=70)

    # Layout: top row = Raw | Masked (side by side); bottom = Ground truth then Model prediction (column)
    try:
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], hspace=0.25, wspace=0.05)
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(rgb)
        ax0.set_title("Raw image", fontsize=12)
        ax0.axis("off")
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(rgb_m)
        ax1.set_title("Masked image", fontsize=12)
        ax1.axis("off")
        ax2 = fig.add_subplot(gs[1, :])
        ax2.imshow(gt_panel, aspect="auto")
        ax2.set_title("Ground truth", fontsize=12)
        ax2.axis("off")
        ax3 = fig.add_subplot(gs[2, :])
        ax3.imshow(pred_panel, aspect="auto")
        ax3.set_title("Model prediction", fontsize=12)
        ax3.axis("off")
        try:
            fig.tight_layout()
        except Exception:
            fig.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.02, hspace=0.2)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        panels = [rgb, rgb_m, gt_panel, pred_panel]
        titles = ["Raw image", "Masked image", "Ground truth", "Model prediction"]
        _save_figure_pil_column(panels, titles, output_path, dpi=150)
    print(f"Saved classification visualization to {output_path}")


def run_segmentation(configs, checkpoint_path, image, label, output_path, device):
    """Run segmentation model on (image, label) and save visualization."""
    configs["device"] = device
    configs["apply_mask_in_infer"] = True
    model = model_utils.create_model(configs)
    ckpt_dir = os.path.dirname(checkpoint_path)
    state_dict_path = checkpoint_path if checkpoint_path.endswith(".pt") else os.path.join(ckpt_dir, "best_model_state_dict.pt")
    if not os.path.isfile(state_dict_path):
        raise FileNotFoundError(f"Checkpoint not found: {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    setattr(model, "testing", True)
    setattr(model, "apply_mask_in_infer", True)
    if hasattr(model, "encoder"):
        setattr(model.encoder, "testing", True)
        setattr(model.encoder, "apply_mask_in_infer", True)

    keys = ["viz_sample_0"]

    with torch.no_grad():
        out, mask_img = model(image, label=label, keys=keys, return_mask=True)

    pred = out.argmax(1)
    masked_image_tensor = build_masked_image(image, mask_img, fill_value=0.0)
    num_classes = configs.get("num_classes", 19)
    segmentation_visualization(image, masked_image_tensor, label, pred, mask_img, num_classes, output_path)


def run_classification(configs, checkpoint_path, image, label, output_path, device):
    """Run classification model on (image, label) and save visualization."""
    configs["device"] = device
    configs["apply_mask_in_infer"] = True
    model = model_utils.create_model(configs)
    ckpt_dir = os.path.dirname(checkpoint_path)
    state_dict_path = checkpoint_path if checkpoint_path.endswith(".pt") else os.path.join(ckpt_dir, "best_model_state_dict.pt")
    if not os.path.isfile(state_dict_path):
        raise FileNotFoundError(f"Checkpoint not found: {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    setattr(model, "testing", True)
    setattr(model, "apply_mask_in_infer", True)

    keys = ["viz_sample_0"]

    # Get mask and prediction: RViT returns (x, hierarchical_mask, final_mask) from forward_mask_in_image_space
    # When hierarchical_features=False, RViT returns (x, None, None) so final_mask can be None
    with torch.no_grad():
        x, hierarchical_mask, final_mask = model.forward_mask_in_image_space(image, keys, label, verbose=False)
    # Run rest of backbone + head
    with torch.no_grad():
        x = model.model.norm_pre(x)
        x = model.model.blocks(x)
        x = model.model.norm(x)
        logits = model.model.forward_head(x)

    # Build masked image for visualization; use final_mask when available
    H_img, W_img = image.shape[2], image.shape[3]
    if final_mask is not None:
        B, num_tokens = final_mask.shape[0], final_mask.shape[1]
        h = w = int(round(num_tokens ** 0.5))
        mask_2d = final_mask.view(B, h, w).float().unsqueeze(1)
        mask_img = F.interpolate(mask_2d, size=(H_img, W_img), mode="nearest").squeeze(1).bool()
    else:
        # Model did not return mask (e.g. hierarchical_features=False): show full image in masked panel
        mask_img = torch.ones(1, H_img, W_img, dtype=torch.bool, device=image.device)
    masked_image_tensor = build_masked_image(image, mask_img.unsqueeze(0), fill_value=0.0)

    # Ground truth and prediction: multilabel (multi-hot) vs single-label
    multilabel = configs.get("multilabel", False)
    if multilabel and label is not None and label.dim() >= 2 and label.shape[1] > 1:
        # Multilabel: GT = indices where label[0] > 0; pred = indices where sigmoid(logits[0]) > 0.5
        gt_classes = (label[0] > 0.5).nonzero(as_tuple=True)[0].cpu().tolist()
        pred_probs = torch.sigmoid(logits[0])
        pred_classes = (pred_probs > 0.5).nonzero(as_tuple=True)[0].cpu().tolist()
    else:
        # Single-label: one GT class, prediction = argmax
        gt_class = label[0].item() if label is not None and label.numel() == 1 else 0
        gt_classes = [gt_class]
        pred_classes = [logits.argmax(1).item()]

    class_names = configs.get("class_names", None)
    classification_visualization(image, masked_image_tensor, gt_classes, pred_classes, class_names, output_path, mask_bool=mask_img)


def main():
    parser = argparse.ArgumentParser(description="Visualize masking at inference")
    parser.add_argument("--task", type=str, required=True, choices=["classification", "segmentation"])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint dir or best_model_state_dict.pt")
    parser.add_argument("--config", type=str, required=True, help="Main config (e.g. configs/main_configs/flair.json)")
    parser.add_argument("--method_config", type=str, required=True, help="Method config (e.g. rvitupernet.json or rvit.json)")
    parser.add_argument("--training_config", type=str, required=True, help="Training config")
    parser.add_argument("--sample_source", type=str, default="webdataset", choices=["path", "webdataset"], help="Load sample from file path or from webdataset")
    parser.add_argument("--sample_path", type=str, default=None, help="Path to input image (.tif); required when sample_source=path")
    parser.add_argument("--label_path", type=str, default=None, help="Path to label (optional; for FLAIR segmentation inferred from sample_path when sample_source=path)")
    parser.add_argument("--webdataset_split", type=str, default="test", choices=["train", "val", "test"], help="Split when sample_source=webdataset")
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index in the split when sample_source=webdataset")
    parser.add_argument("--output_path", type=str, default="visualization_masking.png", help="Output figure path")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("true", "t", "yes", "y", "1"):
            return True
        if v.lower() in ("false", "f", "no", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument("--apply_mask_in_infer", type=str2bool, default=True, help="Use masking for visualization")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone", type=str, default=None, help="Backbone name (e.g. vit_base_patch16_224). If unset, inferred from checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.sample_source == "path":
        if not args.sample_path or not os.path.isfile(args.sample_path):
            print(f"Error: when sample_source=path, sample_path must be an existing file. Got: {args.sample_path}")
            sys.exit(1)

    configs = load_configs_for_viz(args)
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.sample_source == "webdataset":
        print(f"Loading sample from webdataset split={args.webdataset_split} index={args.sample_index} ...")
        image, label = load_sample_from_webdataset(configs, args.webdataset_split, args.sample_index, device)
    else:
        if args.task == "segmentation" and args.label_path is None and "flair" in configs.get("dataset", "").lower():
            root = configs.get("root_path", configs.get("dataset_root_path", ""))
            args.label_path = infer_label_path_from_flair(args.sample_path, root)
        image, label = load_sample_from_path(args.sample_path, args.label_path, configs, device)

    if args.task == "segmentation":
        run_segmentation(configs, args.checkpoint_path, image, label, args.output_path, device)
    else:
        run_classification(configs, args.checkpoint_path, image, label, args.output_path, device)


if __name__ == "__main__":
    main()
