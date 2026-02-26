"""
train_imagenet_synthetic.py
===========================
Two-stage DeepGaze III training on ImageNet synthetic scanpaths.
"""
import argparse
from collections import OrderedDict
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import wandb

# NumPy 2.0 compatibility
if not hasattr(np, 'trapz'): np.trapz = np.trapezoid

from pysaliency.baseline_utils import BaselineModel
from deepgaze_pytorch.layers import Bias, Conv2dMultiInput, FlexibleScanpathHistoryEncoding, LayerNorm, LayerNormMultiInput
from deepgaze_pytorch.modules import DeepGazeIII, FeatureExtractor
from deepgaze_pytorch.features.densenet import RGBDenseNet201
from deepgaze_pytorch.data import ImageDataset, ImageDatasetSampler, FixationDataset, FixationMaskTransform
from deepgaze_pytorch.training import _train
from parquet_to_pysaliency import parquet_to_pysaliency

# --- Config ---
PARQUET_PATH   = "/mnt/lustre-grete/usr/u13879/scanpather/imagenet_subset/2026_02_25_18_07_18_fb1c5cce/scanpaths/merged.parquet"
OUT_DIR        = "./runs/run_01"
DEVICE         = "cuda"
BATCH_SIZE     = 4
TRAIN_FRAC     = 0.9
CLASS_FRAC     = 0.05
SEED           = 3141
IMAGE_BASE_DIR = "/mnt/lustre-grete/usr/u13879/datasets/ImageNet/train_images"
IMAGENET_SIZE  = 224  
NUM_EPOCHS     = 10
LR             = 1e-3

# --- Helper: Resizing Logic ---
# This ensures images are standardized after loading from files/LMDB
standard_resize = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])

_fixation_mask_transform = FixationMaskTransform(sparse=False)

def resize_transform(data):
    """
    Resize every image and centerbias to 224×224 and convert to float32,
    then produce a fixed-shape dense fixation_mask.

    Images arriving from LMDB are raw numpy arrays (C, H, W) uint8 at their
    original ImageNet sizes (highly variable).  Images from FileStimuli arrive
    as PIL Images.  Both must be resized so that all items in a batch share the
    same spatial dimensions, which is required for torch's collate_fn.
    The centerbias is stored at the original image resolution too and must
    be resized to match.
    """
    img = data['image']

    if isinstance(img, Image.Image):
        # FileStimuli path: PIL Image → resize → (H, W, C) uint8 numpy
        img = np.array(standard_resize(img))           # (224, 224, 3)
        data['image'] = img.transpose(2, 0, 1).astype(np.float32)  # (3, 224, 224)
    elif isinstance(img, np.ndarray):
        # LMDB path: already (C, H, W) uint8, but at original ImageNet size.
        # Convert back to PIL to use the same resize pipeline, then to float32.
        pil_img = Image.fromarray(img.transpose(1, 2, 0))  # (H, W, C)
        img_resized = np.array(standard_resize(pil_img))   # (224, 224, 3)
        data['image'] = img_resized.transpose(2, 0, 1).astype(np.float32)  # (3, 224, 224)

    # Resize centerbias from original image size to 224×224.
    cb = data['centerbias']  # (H_orig, W_orig) float
    if cb.shape != (IMAGENET_SIZE, IMAGENET_SIZE):
        cb_tensor = torch.from_numpy(cb.astype(np.float32))[None, None]  # (1,1,H,W)
        cb_resized = torch.nn.functional.interpolate(
            cb_tensor, size=(IMAGENET_SIZE, IMAGENET_SIZE), mode='bilinear', align_corners=False
        )[0, 0]
        data['centerbias'] = cb_resized.numpy()

    # Convert x/y coordinate arrays → dense fixation_mask tensor (H×W).
    return _fixation_mask_transform(data)

# --- Model Builders ---
def build_saliency_network(input_channels: int = 2048):
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNorm(input_channels)),
        ('conv0', nn.Conv2d(input_channels, 8, (1, 1), bias=False)),
        ('bias0', Bias(8)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(8)),
        ('conv1', nn.Conv2d(8, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
        ('layernorm2', LayerNorm(16)),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
        ('bias2', Bias(1)),
        ('softplus2', nn.Softplus()),
    ]))

def build_scanpath_network():
    return nn.Sequential(OrderedDict([
        ('encoding0', FlexibleScanpathHistoryEncoding(in_fixations=4, channels_per_fixation=3, out_channels=128, kernel_size=[1, 1], bias=True)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
    ]))

def build_fixation_selection_network(scanpath_features: int = 16):
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput([1, scanpath_features])),
        ('conv0', Conv2dMultiInput([1, scanpath_features], 128, (1, 1), bias=False)),
        ('bias0', Bias(128)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
    ]))

# --- Loaders ---
def make_spatial_loader(stimuli, fixations, centerbias, batch_size, cache_path):
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset = ImageDataset(
        stimuli=stimuli, fixations=fixations, centerbias_model=centerbias,
        transform=resize_transform, 
        average='image', lmdb_path=str(cache_path),
    )
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size),
        pin_memory=False, num_workers=4,
    )

def make_scanpath_loader(stimuli, fixations, centerbias, batch_size, cache_path):
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset = FixationDataset(
        stimuli=stimuli, fixations=fixations, centerbias_model=centerbias,
        included_fixations=[-1, -2, -3, -4], allow_missing_fixations=True,
        transform=resize_transform, 
        average='image', lmdb_path=str(cache_path),
    )
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size),
        pin_memory=False, num_workers=4,
    )

def make_scheduler(optimizer, num_epochs):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[num_epochs], gamma=1e-5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', default=PARQUET_PATH)
    parser.add_argument('--out_dir', default=OUT_DIR)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--class_frac', type=float, default=CLASS_FRAC)
    parser.add_argument('--lr', type=float, default=LR)
    args = parser.parse_args()

    out_dir, device = Path(args.out_dir), "cuda"
    cache_dir = out_dir / 'lmdb_cache'

    # Load data
    train_stimuli, train_fixations, val_stimuli, val_fixations = parquet_to_pysaliency(
        args.parquet, class_frac=args.class_frac, image_base_dir=IMAGE_BASE_DIR, target_size=IMAGENET_SIZE,
    )

    # Dummy Centerbias to avoid disk I/O on shapes
    class ZeroCenterbias:
        def log_density(self, stimulus): return np.zeros((224, 224))
    centerbias, train_ll, val_ll = ZeroCenterbias(), 0.0, 0.0

    # Stage 1
    print("\n=== Stage 1: spatial model ===")
    model = DeepGazeIII(
        features=FeatureExtractor(RGBDenseNet201(), [
            '1.features.denseblock4.denselayer32.norm1',
            '1.features.denseblock4.denselayer32.conv1',
            '1.features.denseblock4.denselayer31.conv2',
        ]),
        saliency_network=build_saliency_network(2048),
        scanpath_network=None,
        fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
        downsample=2, readout_factor=4, saliency_map_factor=4, included_fixations=[],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, NUM_EPOCHS)
    
    t_loader = make_spatial_loader(train_stimuli, train_fixations, centerbias, args.batch_size, cache_dir / 'spatial_train')
    v_loader = make_spatial_loader(val_stimuli, val_fixations, centerbias, args.batch_size, cache_dir / 'spatial_val')

    _train(out_dir / 'spatial', model, t_loader, train_ll, v_loader, val_ll, 
           optimizer, scheduler, minimum_learning_rate=1e-7, device=device)

    # Stage 2
    print("\n=== Stage 2: scanpath model ===")
    model = DeepGazeIII(
        features=FeatureExtractor(RGBDenseNet201(), [
            '1.features.denseblock4.denselayer32.norm1',
            '1.features.denseblock4.denselayer32.conv1',
            '1.features.denseblock4.denselayer31.conv2',
        ]),
        saliency_network=build_saliency_network(2048),
        scanpath_network=build_scanpath_network(),
        fixation_selection_network=build_fixation_selection_network(scanpath_features=16),
        downsample=2, readout_factor=4, saliency_map_factor=4, included_fixations=[-1, -2, -3, -4],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, NUM_EPOCHS)

    t_loader = make_scanpath_loader(train_stimuli, train_fixations, centerbias, args.batch_size, cache_dir / 'scanpath_train')
    v_loader = make_scanpath_loader(val_stimuli, val_fixations, centerbias, args.batch_size, cache_dir / 'scanpath_val')

    _train(out_dir / 'scanpath_full', model, t_loader, train_ll, v_loader, val_ll, 
           optimizer, scheduler, minimum_learning_rate=1e-7, device=device, startwith=out_dir / 'spatial' / 'final.pth')

if __name__ == '__main__':
    main()