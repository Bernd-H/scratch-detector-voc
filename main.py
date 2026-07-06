# -*- coding: utf-8 -*-
"""
# 192.151 Introduction to Deep Learning 2026S
## Project 2 — Object Detection: A YOLO-Style Detector Trained From Scratch on Pascal VOC

**Pipeline overview:**
1. **Dataset**: Pascal VOC 2012 (downloaded via `torchvision`)
2. **CNN Backbone**: Stack of Residual Blocks (5 downsampling stages → 13×13 feature map)
3. **Detection Head**: Predicts class logits + box offsets for 5 anchors per grid cell
4. **Loss**: Objectness (BCE) + Classification (BCE + label-smoothing) + Localisation (SmoothL1) + No-object penalty
5. **Post-processing**: Per-class Non-Maximum Suppression (NMS)
6. **Evaluation**: mAP@0.50 (VOC 11-point interpolation)
7. **Visualisation**: Draw predicted boxes on val images
"""

# Install / verify dependencies (run once)

import sys
!{sys.executable} -m pip install \
    matplotlib \
    scikit-learn \
    pandas \
    scipy \
    tqdm \
    pillow \
    opencv-python

# Output directory for checkpoints/plots.
import os
OUTPUT_DIR = os.path.expanduser('~/workspace/outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'Outputs will be saved to: {OUTPUT_DIR}')

import os, random, math, time, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import datasets
from torch.utils.data import WeightedRandomSampler


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import xml.etree.ElementTree as ET
from PIL import Image

# ── Reproducibility vs. speed ────────────────────────────────────────────────
# cudnn.deterministic and cudnn.benchmark are mutually exclusive in effect:
# deterministic forces a fixed (reproducible but not necessarily fastest)
# conv algorithm; benchmark profiles all algorithms on first use and picks
# the fastest for your exact input shape (free win since IMG_SIZE/BATCH_SIZE
# are fixed here), at the cost of run-to-run bitwise reproducibility.
# Set REPRODUCIBLE=True if you need exact repeatability for debugging;
# leave False for max throughput.
REPRODUCIBLE = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = REPRODUCIBLE
torch.backends.cudnn.benchmark = not REPRODUCIBLE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# ── Global Hyper-parameters ──────────────────────────────────────────────────

IMG_SIZE    = 416          # input resolution (square)
GRID_SIZE   = 13           # feature-map grid = IMG_SIZE // 32

# Five anchors chosen by k-means clustering on VOC 2012 GT boxes
# Format: (width, height) in grid-cell units
ANCHORS = [
    (1.08, 1.19),
    (3.42, 4.41),
    (6.63, 11.38),
    (9.42, 5.11),
    (16.62, 10.52),
]
NUM_ANCHORS = len(ANCHORS)

# Training schedule
# LR scaled with BATCH_SIZE (linear scaling rule) since larger batches need
# a proportionally larger LR to take equivalently-sized steps per epoch.
BATCH_SIZE   = 64
NUM_EPOCHS   = 80
LR           = 3e-4 * (BATCH_SIZE / 16)
WEIGHT_DECAY = 1e-4

# Loss weights (lambda values — YOLO-style)
LAMBDA_OBJ   = 1.0
LAMBDA_NOOBJ = 1.0
LAMBDA_COORD = 5.0
LAMBDA_CLS   = 1.0

# Label smoothing applied inside Dataset targets
LABEL_SMOOTHING = 0.1

# Inference thresholds
CONF_THRESH    = 0.45
NMS_IOU_THRESH = 0.35

# Separate, much lower threshold for mAP evaluation: AP integrates precision
# over the FULL recall range, so eval must keep low-confidence boxes too.
# CONF_THRESH (0.25) is only for the human-facing visualisation.
EVAL_CONF_THRESH = 0.001

# mAP evaluation IoU threshold
MAP_IOU_THRESH = 0.50

# VOC_CLASSES = [
#     'aeroplane','bicycle','bird','boat','bottle',
#     'bus','car','cat','chair','cow',
#     'diningtable','dog','horse','motorbike','person',
#     'pottedplant','sheep','sofa','train','tvmonitor',
# ]

VOC_CLASSES = [
    "aeroplane", "bicycle",
    "bus", "car", "motorbike",
]

NUM_CLASSES = len(VOC_CLASSES)

print('Config loaded.')

class_instance_counts = {
    'car': 1191, 'aeroplane': 470,
    'bicycle': 410, 'motorbike': 375, 'bus': 317,
}
counts_tensor = torch.tensor(
    [class_instance_counts[c] for c in VOC_CLASSES], dtype=torch.float32
)
raw_weights = counts_tensor.sum() / (NUM_CLASSES * counts_tensor)
class_weights = raw_weights.sqrt().to(DEVICE)

print(dict(zip(VOC_CLASSES, class_weights.tolist())))

# ── 4. Dataset Preparation & Preprocessing ───────────────────────────────────
# Downloads Pascal VOC 2012 via torchvision.datasets.VOCDetection (downloads
# + extracts the official ~2GB tarball into VOC_ROOT/VOCdevkit/VOC2012/ on
# first run, then reuses the cached copy on subsequent runs).
# Each image is resized to IMG_SIZE×IMG_SIZE; annotations are converted to
# YOLO grid format: (tx, ty, tw, th) per assigned anchor.

VOC_ROOT = os.path.expanduser('~/data')
os.makedirs(VOC_ROOT, exist_ok=True)

print('Downloading Pascal VOC 2012 (skipped automatically if already present)...')
_ = datasets.VOCDetection(root=VOC_ROOT, year='2012', image_set='train', download=True)
print('VOC 2012 ready.')

DATA_ROOT = os.path.join(VOC_ROOT, 'VOCdevkit', 'VOC2012')

# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = T.Compose([
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Anchor utilities ──────────────────────────────────────────────────────────
_anchors_t = torch.tensor(ANCHORS, dtype=torch.float32)  # (A, 2) grid units
IGNORE_IOU_THRESH = 0.5


def iou_wh(wh1, wh2):
    """Shape-based IoU (shared centre) between a single box wh1 and N boxes wh2."""
    inter_w = torch.min(wh1[..., 0], wh2[..., 0])
    inter_h = torch.min(wh1[..., 1], wh2[..., 1])
    inter   = inter_w * inter_h
    union   = wh1[..., 0]*wh1[..., 1] + wh2[..., 0]*wh2[..., 1] - inter
    return inter / (union + 1e-6)

def best_anchor_for(gt_wh_grid):
    """Return anchor index that best matches gt box by WH-IoU."""
    ious = iou_wh(_anchors_t, gt_wh_grid.unsqueeze(0))
    return int(ious.argmax())


def random_resized_crop(img, objects, orig_w, orig_h, scale=(0.7, 1.0)):
    """Randomly crop a sub-region and rescale boxes accordingly. Returns
    new PIL image (still orig size) and adjusted object list (in orig_w/orig_h space)."""
    area = orig_w * orig_h
    target_area = random.uniform(*scale) * area
    aspect = random.uniform(0.8, 1.25)
    w = int(round(math.sqrt(target_area * aspect)))
    h = int(round(math.sqrt(target_area / aspect)))
    w = min(w, orig_w); h = min(h, orig_h)

    x0 = random.randint(0, orig_w - w)
    y0 = random.randint(0, orig_h - h)

    cropped = img.crop((x0, y0, x0 + w, y0 + h)).resize((orig_w, orig_h))

    new_objects = []
    for obj in objects:
        bb = obj['bndbox']
        xmin = float(bb['xmin']) - x0
        ymin = float(bb['ymin']) - y0
        xmax = float(bb['xmax']) - x0
        ymax = float(bb['ymax']) - y0
        # clip to crop window
        xmin = max(0, min(xmin, w)); xmax = max(0, min(xmax, w))
        ymin = max(0, min(ymin, h)); ymax = max(0, min(ymax, h))
        # drop boxes that vanish or become degenerate after cropping
        if xmax - xmin < 4 or ymax - ymin < 4:
            continue
        # rescale into orig_w/orig_h coordinate space (since we resized crop back up)
        scale_x = orig_w / w
        scale_y = orig_h / h
        new_bb = {
            'xmin': str(xmin * scale_x), 'ymin': str(ymin * scale_y),
            'xmax': str(xmax * scale_x), 'ymax': str(ymax * scale_y),
        }
        new_obj = dict(obj)
        new_obj['bndbox'] = new_bb
        new_objects.append(new_obj)
    return cropped, new_objects

# ── Dataset class ─────────────────────────────────────────────────────────────


class VOCDetectionDataset(Dataset):
    def __init__(self, root, year='2012', image_set='train', transform=None):
        self.root = root
        self.transform = transform
        self.img_dir = os.path.join(root, 'JPEGImages')
        self.ann_dir = os.path.join(root, 'Annotations')
        self.is_train = image_set == 'train'

        split_file = os.path.join(root, 'ImageSets', 'Main', f'{image_set}.txt')
        with open(split_file) as f:
            ids = [line.strip() for line in f if line.strip()]

        self.samples = []
        for img_id in ids:
            ann_path = os.path.join(self.ann_dir, f'{img_id}.xml')
            objs = self._parse_annotation(ann_path)
            if any(o['name'] in VOC_CLASSES for o in objs):
                self.samples.append((img_id, objs))

    @staticmethod
    def _parse_annotation(ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        size = root.find('size')
        orig_w = int(size.find('width').text)
        orig_h = int(size.find('height').text)
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bb = obj.find('bndbox')
            diff_node = obj.find('difficult')
            difficult = bool(int(diff_node.text)) if diff_node is not None else False
            objects.append({
                'name': name, 'orig_w': orig_w, 'orig_h': orig_h,
                'difficult': difficult,
                'bndbox': {
                    'xmin': bb.find('xmin').text, 'ymin': bb.find('ymin').text,
                    'xmax': bb.find('xmax').text, 'ymax': bb.find('ymax').text,
                }
            })
        return objects

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, objects = self.samples[idx]
        img_path = os.path.join(self.img_dir, f'{img_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        if self.is_train and random.random() < 0.5:
            img, objects = random_resized_crop(img, objects, orig_w, orig_h)
            if not any(o['name'] in VOC_CLASSES for o in objects):
                img = Image.open(img_path).convert('RGB')
                objects = self.samples[idx][1]

        img = img.resize((IMG_SIZE, IMG_SIZE))

        target = torch.zeros(GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + NUM_CLASSES)
        ignore = torch.zeros(GRID_SIZE, GRID_SIZE, NUM_ANCHORS, dtype=torch.bool)

        # Raw GT for evaluation: one entry per object, independent of any
        # grid/anchor collisions in the encoded `target` tensor above, and
        # carrying the VOC `difficult` flag for the official eval protocol.
        gt_boxes = []      # list of [x1, y1, x2, y2] in IMG_SIZE pixel space
        gt_labels = []     # list of class indices
        gt_difficult = []  # list of bool

        for obj in objects:
            cls_name = obj['name']
            if cls_name not in VOC_CLASSES:
                continue
            cls_idx = VOC_CLASSES.index(cls_name)

            bb   = obj['bndbox']
            xmin = float(bb['xmin']) / orig_w * IMG_SIZE
            ymin = float(bb['ymin']) / orig_h * IMG_SIZE
            xmax = float(bb['xmax']) / orig_w * IMG_SIZE
            ymax = float(bb['ymax']) / orig_h * IMG_SIZE

            gt_boxes.append([xmin, ymin, xmax, ymax])
            gt_labels.append(cls_idx)
            gt_difficult.append(bool(obj.get('difficult', False)))

            cell = IMG_SIZE / GRID_SIZE
            cx   = (xmin + xmax) / 2.0 / cell
            cy   = (ymin + ymax) / 2.0 / cell
            bw   = (xmax - xmin) / cell
            bh   = (ymax - ymin) / cell

            gx = min(int(cx), GRID_SIZE - 1)
            gy = min(int(cy), GRID_SIZE - 1)

            ious = iou_wh(_anchors_t, torch.tensor([bw, bh]).unsqueeze(0))  # (A,)
            best_a = int(ious.argmax())

            # mark non-best anchors in this cell with decent overlap as "ignore"
            for a in range(NUM_ANCHORS):
                if a != best_a and ious[a] > IGNORE_IOU_THRESH:
                    ignore[gy, gx, a] = True

            tx = cx - gx
            ty = cy - gy
            tw = math.log(bw / ANCHORS[best_a][0] + 1e-6)
            th = math.log(bh / ANCHORS[best_a][1] + 1e-6)

            eps = LABEL_SMOOTHING
            cls_vec = torch.full((NUM_CLASSES,), eps / NUM_CLASSES)
            cls_vec[cls_idx] = 1.0 - eps + eps / NUM_CLASSES

            target[gy, gx, best_a, 0]  = 1.0
            target[gy, gx, best_a, 1]  = tx
            target[gy, gx, best_a, 2]  = ty
            target[gy, gx, best_a, 3]  = tw
            target[gy, gx, best_a, 4]  = th
            target[gy, gx, best_a, 5:] = cls_vec
            ignore[gy, gx, best_a] = False   # never ignore a confirmed positive

        # ── Horizontal flip (train only) — must flip target, ignore, AND raw GT ──
        if self.is_train and random.random() < 0.5:
            img = TF.hflip(img)
            flipped_target = torch.zeros_like(target)
            flipped_ignore  = torch.zeros_like(ignore)
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    new_gx = GRID_SIZE - 1 - gx
                    cell_data = target[gy, gx]
                    if cell_data[..., 0].any():
                        flipped_cell = cell_data.clone()
                        flipped_cell[..., 1] = 1.0 - cell_data[..., 1]
                        flipped_target[gy, new_gx] = flipped_cell
                    flipped_ignore[gy, new_gx] = ignore[gy, gx]
            target = flipped_target
            ignore = flipped_ignore

            flipped_gt_boxes = []
            for x1, y1, x2, y2 in gt_boxes:
                flipped_gt_boxes.append([IMG_SIZE - x2, y1, IMG_SIZE - x1, y2])
            gt_boxes = flipped_gt_boxes

        if self.transform:
            img = self.transform(img)

        gt = {
            'boxes': torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes
                     else torch.zeros(0, 4, dtype=torch.float32),
            'labels': torch.tensor(gt_labels, dtype=torch.long) if gt_labels
                      else torch.zeros(0, dtype=torch.long),
            'difficult': torch.tensor(gt_difficult, dtype=torch.bool) if gt_difficult
                         else torch.zeros(0, dtype=torch.bool),
        }

        return img, target, ignore, gt


# ── Build DataLoaders ─────────────────────────────────────────────────────────

train_ds = VOCDetectionDataset(DATA_ROOT, image_set='train', transform=train_transform)
val_ds   = VOCDetectionDataset(DATA_ROOT, image_set='val',   transform=val_transform)

def class_presence_weight(objects, class_freq_inv):
    present = [c for c in VOC_CLASSES if any(o['name'] == c for o in objects)]
    if not present:
        return 1.0
    return max(class_freq_inv[c] for c in present)

class_freq_inv = {c: 1.0 / class_instance_counts[c] for c in VOC_CLASSES}
sample_weights = [
    class_presence_weight(objs, class_freq_inv) for _, objs in train_ds.samples
]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)


def detection_collate(batch):
    """Custom collate: imgs/target/ignore stack normally (fixed shape per
    sample); gt dicts have variable-length boxes/labels/difficult, so they
    stay as a plain list of dicts, one per sample in the batch."""
    imgs, targets, ignores, gts = zip(*batch)
    imgs    = torch.stack(imgs, dim=0)
    targets = torch.stack(targets, dim=0)
    ignores = torch.stack(ignores, dim=0)
    return imgs, targets, ignores, list(gts)


# persistent_workers avoids respawning the worker pool every epoch.
# prefetch_factor=4 lets workers stay a few batches ahead of the GPU.
NUM_WORKERS = 8

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          collate_fn=detection_collate,
                          persistent_workers=True, prefetch_factor=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                          collate_fn=detection_collate,
                          persistent_workers=True, prefetch_factor=4)

print(f'Train: {len(train_ds):,} images   Val: {len(val_ds):,} images')

from sklearn.cluster import KMeans

def collect_wh(dataset):
    whs = []
    for img_id, objs in dataset.samples:
        for o in objs:
            if o['name'] not in VOC_CLASSES: continue
            bb = o['bndbox']
            w = (float(bb['xmax'])-float(bb['xmin'])) / o['orig_w'] * GRID_SIZE
            h = (float(bb['ymax'])-float(bb['ymin'])) / o['orig_h'] * GRID_SIZE
            whs.append([w, h])
    return np.array(whs)

wh = collect_wh(train_ds)
km = KMeans(n_clusters=5, random_state=SEED).fit(wh)
print(sorted(km.cluster_centers_.tolist(), key=lambda x: x[0]*x[1]))

# ── 5. Model Architecture ─────────────────────────────────────────────────────
# ResNet-style backbone + 2-layer convolutional detection head.
# Total stride = 32 (416 → 13 grid). Backbone produces (B, 512, 13, 13) features.

class ResBlock(nn.Module):
    """
    Standard residual block: Conv-BN-LReLU → Conv-BN → skip add → LReLU.
    A 1×1 projection is used when channel count or stride changes.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        return F.leaky_relu(out + self.skip(x), 0.1)


class ScratchDetectorBackbone(nn.Module):
    """
    Five-stage CNN backbone.
    Input  : (B, 3,   416, 416)
    Output : (B, 512,  13,  13)

    Stem   : 3→64,   7×7 conv, stride 2 + MaxPool  → 104×104
    Stage1 : 64→128,  stride 2  (1 ResBlock)        →  52×52
    Stage2 : 128→256, stride 2  (2 ResBlocks)       →  26×26
    Stage3 : 256→512, stride 2  (2 ResBlocks)       →  13×13
    Stage4 : 512→512, stride 1  (2 ResBlocks)       →  13×13  (refine)
    """
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = self._stage(64,  128, n=1, stride=2)
        self.stage2 = self._stage(128, 256, n=2, stride=2)
        self.stage3 = self._stage(256, 512, n=2, stride=2)
        self.stage4 = self._stage(512, 512, n=2, stride=1)

    @staticmethod
    def _stage(in_ch, out_ch, n, stride):
        layers = [ResBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n - 1):
            layers.append(ResBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x   # (B, 512, 13, 13)


class ScratchDetectorHead(nn.Module):
    """
    Detection head: maps (B, 512, 13, 13) → (B, G, G, A, 5+C).

    Per anchor output:
        [..., 0]   : raw objectness logit  (σ → P(object))
        [..., 1:3] : (tx, ty)  centre offset within grid cell  (σ applied at decode)
        [..., 3:5] : (tw, th)  log-scale correction relative to anchor size
        [..., 5:]  : class logits  (C values)
    """
    def __init__(self):
        super().__init__()
        mid = 256
        self.conv1  = nn.Conv2d(512, mid, 3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(mid)
        self.conv2  = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(mid)
        self.output = nn.Conv2d(mid, NUM_ANCHORS * (5 + NUM_CLASSES), 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = self.output(x)                          # (B, A*(5+C), G, G)
        B, _, G, _ = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()     # (B, G, G, A*(5+C))
        x = x.view(B, G, G, NUM_ANCHORS, 5 + NUM_CLASSES)
        return x


class ScratchDetector(nn.Module):
    """Complete detector: backbone + head."""
    def __init__(self):
        super().__init__()
        self.backbone = ScratchDetectorBackbone()
        self.head     = ScratchDetectorHead()

    def forward(self, x):
        return self.head(self.backbone(x))


# ── Sanity check ─────────────────────────────────────────────────────────────
model = ScratchDetector().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f'ScratchDetector | parameters: {n_params:,}')

_x = torch.zeros(2, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
_y = model(_x)
print(f'Output shape : {tuple(_y.shape)}  '
      f'expected (2, {GRID_SIZE}, {GRID_SIZE}, {NUM_ANCHORS}, {5+NUM_CLASSES})')
del _x, _y

# torch.compile fuses kernels (conv+bn+activation, etc.) for a meaningful
# free speedup on a fixed-shape training loop. Wrapped in try/except since
# compile support depends on the exact torch/triton/CUDA versions available;
# if it's unavailable or fails, training still works uncompiled.
if DEVICE.type == 'cuda':
    try:
        model = torch.compile(model)
        print('torch.compile enabled.')
    except Exception as e:
        print(f'torch.compile unavailable ({e}); continuing uncompiled.')

# Keep a handle to the *uncompiled* module for state_dict save/load.
# torch.compile wraps the model in an OptimizedModule whose state_dict keys
# may be prefixed (e.g. '_orig_mod.xxx'); saving/loading through this raw
# reference keeps checkpoints portable to any environment, compiled or not.
model_raw = model._orig_mod if hasattr(model, '_orig_mod') else model

# ── 6. Detection Loss ─────────────────────────────────────────────────────────
# Four-term YOLO-style loss:
#
#   L = λ_obj   * BCE(σ(logit), 1)           [cells WITH an object]
#     + λ_noobj * BCE(σ(logit), 0)           [cells WITHOUT an object]
#     + λ_coord * SmoothL1(pred_box, gt_box) [cells WITH an object]
#     + λ_cls   * BCE(pred_cls, gt_cls_soft) [cells WITH an object]
#
# Class targets already carry label-smoothing from the Dataset,
# so we use BCEWithLogitsLoss (not CrossEntropyLoss) for the class term.

LAMBDA_CLS_BG = 0.5   # background class-suppression weight

class DetectionLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.bce      = nn.BCEWithLogitsLoss(reduction='sum')
        self.bce_cls  = nn.BCEWithLogitsLoss(reduction='sum', weight=class_weights)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')

    def forward(self, preds, targets, ignore):
        obj_mask   = targets[..., 0] == 1
        noobj_mask = (~obj_mask) & (~ignore)
        N_pos   = obj_mask.sum().clamp(min=1).float()
        N_noobj = noobj_mask.sum().clamp(min=1).float()

        loss_obj   = LAMBDA_OBJ   * self.bce(preds[..., 0][obj_mask],
                                              targets[..., 0][obj_mask]) / N_pos
        loss_noobj = LAMBDA_NOOBJ * self.bce(preds[..., 0][noobj_mask],
                                              targets[..., 0][noobj_mask]) / N_noobj
        loss_coord = LAMBDA_COORD * self.smooth_l1(
            preds[..., 1:5][obj_mask], targets[..., 1:5][obj_mask]) / N_pos

        loss_cls = LAMBDA_CLS * self.bce_cls(
            preds[..., 5:][obj_mask], targets[..., 5:][obj_mask]) / N_pos

        loss_cls_bg = LAMBDA_CLS_BG * self.bce(
            preds[..., 5:][noobj_mask], torch.zeros_like(preds[..., 5:][noobj_mask])
        ) / N_noobj

        total = loss_obj + loss_noobj + loss_coord + loss_cls + loss_cls_bg
        return total, {'obj': loss_obj.item(), 'noobj': loss_noobj.item(),
                       'coord': loss_coord.item(), 'cls': loss_cls.item(),
                       'cls_bg': loss_cls_bg.item()}

criterion = DetectionLoss(class_weights)
print('DetectionLoss ready.')

# ── 7. Decoding Predictions & Non-Maximum Suppression (NMS) ──────────────────
# The model outputs raw (tx, ty, tw, th) for each anchor.
# Decoding converts these to absolute pixel coordinates (x1, y1, x2, y2).
# NMS then suppresses duplicate detections per class.

CELL_SIZE    = IMG_SIZE / GRID_SIZE                         # pixels per grid cell
_anchors_dev = torch.tensor(ANCHORS, dtype=torch.float32)  # (A, 2), moved to device later


def decode_predictions(raw_preds):
    """
    Decode raw model output for ONE image to pixel-space boxes.

    Args:
        raw_preds : (G, G, A, 5+C) — output for a single image

    Returns:
        boxes  : (G*G*A, 4) tensor  [x1, y1, x2, y2] in pixels
        scores : (G*G*A,)  tensor   objectness * max-class probability
        labels : (G*G*A,)  long     argmax class index
    """
    G      = GRID_SIZE
    device = raw_preds.device
    anch   = _anchors_dev.to(device)          # (A, 2)

    # Grid offsets: gx_grid[gy, gx] = gx,  gy_grid[gy, gx] = gy
    gy_idx   = torch.arange(G, device=device, dtype=torch.float32)
    gx_idx   = torch.arange(G, device=device, dtype=torch.float32)
    gy_grid, gx_grid = torch.meshgrid(gy_idx, gx_idx, indexing='ij')  # (G, G)

    # Objectness and class probabilities
    obj_score = torch.sigmoid(raw_preds[..., 0])              # (G, G, A)
    cls_prob  = torch.sigmoid(raw_preds[..., 5:])
    cls_score, cls_label = cls_prob.max(dim=-1)               # (G, G, A)
    conf_score = obj_score * cls_score

    # Box decoding
    tx = torch.sigmoid(raw_preds[..., 1])                     # (G, G, A)
    ty = torch.sigmoid(raw_preds[..., 2])
    tw = raw_preds[..., 3]
    th = raw_preds[..., 4]

    cx = (gx_grid.unsqueeze(-1) + tx) * CELL_SIZE             # pixels
    cy = (gy_grid.unsqueeze(-1) + ty) * CELL_SIZE
    bw = anch[:, 0].view(1, 1, -1) * torch.exp(tw.clamp(-4, 4)) * CELL_SIZE
    bh = anch[:, 1].view(1, 1, -1) * torch.exp(th.clamp(-4, 4)) * CELL_SIZE

    x1 = (cx - bw / 2).clamp(0, IMG_SIZE)
    y1 = (cy - bh / 2).clamp(0, IMG_SIZE)
    x2 = (cx + bw / 2).clamp(0, IMG_SIZE)
    y2 = (cy + bh / 2).clamp(0, IMG_SIZE)

    boxes  = torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)
    scores = conf_score.view(-1)
    labels = cls_label.view(-1)
    return boxes, scores, labels


def nms_per_class(boxes, scores, labels,
                  conf_thresh=CONF_THRESH, iou_thresh=NMS_IOU_THRESH):
    """
    Confidence threshold + per-class NMS.

    Args:
        boxes  : (N, 4)  x1y1x2y2
        scores : (N,)
        labels : (N,) long

    Returns kept boxes, scores, labels.
    """
    keep = scores >= conf_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.numel() == 0:
        return boxes, scores, labels

    out_b, out_s, out_l = [], [], []
    for cls_id in labels.unique():
        m = labels == cls_id
        kept_idx = torchvision.ops.nms(boxes[m].float(), scores[m].float(), iou_thresh)
        out_b.append(boxes[m][kept_idx])
        out_s.append(scores[m][kept_idx])
        out_l.append(labels[m][kept_idx])

    return torch.cat(out_b), torch.cat(out_s), torch.cat(out_l)

print('Decode + NMS utilities ready.')

# ── 8. mAP@0.50 Evaluation ───────────────────────────────────────────────────
# Standard VOC 11-point interpolated mAP.
# We accumulate (score, is_tp) per class over the val set, then compute AP
# from the sorted precision-recall curve.

def box_iou_single(b1, b2):
    """IoU between box b1 (4,) and a set of boxes b2 (N, 4). Returns (N,)."""
    ix1 = torch.max(b1[0], b2[:, 0])
    iy1 = torch.max(b1[1], b2[:, 1])
    ix2 = torch.min(b1[2], b2[:, 2])
    iy2 = torch.min(b1[3], b2[:, 3])
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[:,2]-b2[:,0]) * (b2[:,3]-b2[:,1])
    return inter / (a1 + a2 - inter + 1e-6)


def compute_ap(rec, prec):
    """VOC 11-point interpolated AP."""
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        mask = rec >= t
        if mask.any():
            ap += prec[mask].max().item()
    return ap / 11.0


def _organize_gt(gt_single):
    """
    Organise one image's raw GT dict {'boxes','labels','difficult'} by class.

    Returns:
        gt_list   : {cls_id: Tensor (N_c, 4)}        all GT boxes per class
        diff_list : {cls_id: Tensor (N_c,) bool}     parallel difficult flags
    """
    gt_list, diff_list = {}, {}
    boxes = gt_single['boxes']
    labels = gt_single['labels']
    difficult = gt_single['difficult']
    for cls_id in labels.unique().tolist():
        m = labels == cls_id
        gt_list[cls_id] = boxes[m]
        diff_list[cls_id] = difficult[m]
    return gt_list, diff_list


@torch.no_grad()
def evaluate_map(model, loader, iou_thresh=MAP_IOU_THRESH, max_batches=None):
    """Compute mAP@iou_thresh over a DataLoader.

    Follows the VOC evaluation protocol (Everingham et al.): GT boxes
    flagged `difficult` are excluded from the GT count (they don't
    contribute to recall), and any detection whose best match is a
    difficult GT box is simply discarded — neither a true positive nor
    a false positive — rather than penalised.
    """
    model.eval()
    class_preds = {c: [] for c in range(NUM_CLASSES)}
    class_n_gt  = {c: 0  for c in range(NUM_CLASSES)}

    for batch_idx, (imgs, targets, ignore, gts) in enumerate(
            tqdm(loader, desc='mAP eval', leave=False)):
        if max_batches and batch_idx >= max_batches:
            break
        imgs    = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            preds = model(imgs)

        for i in range(imgs.shape[0]):
            boxes, scores, labels = decode_predictions(preds[i])
            boxes, scores, labels = nms_per_class(boxes, scores, labels,
                                                   conf_thresh=EVAL_CONF_THRESH)

            gt_single = {k: v.to(DEVICE) for k, v in gts[i].items()}
            gt_list, diff_list = _organize_gt(gt_single)
            for cls_, gt_b in gt_list.items():
                class_n_gt[cls_] += int((~diff_list[cls_]).sum().item())

            matched = {c: [False] * len(v) for c, v in gt_list.items()}
            for k in range(len(boxes)):
                cls_id = int(labels[k].item())
                score  = float(scores[k].item())
                if cls_id not in gt_list or len(gt_list[cls_id]) == 0:
                    class_preds[cls_id].append((score, 0))
                    continue
                gt_boxes_c = gt_list[cls_id]
                ious = box_iou_single(boxes[k], gt_boxes_c)
                best_iou, best_j = ious.max(0)
                best_j = best_j.item()
                if best_iou >= iou_thresh:
                    if diff_list[cls_id][best_j]:
                        continue  # matched a difficult GT: ignore detection entirely
                    if not matched[cls_id][best_j]:
                        matched[cls_id][best_j] = True
                        class_preds[cls_id].append((score, 1))
                    else:
                        class_preds[cls_id].append((score, 0))
                else:
                    class_preds[cls_id].append((score, 0))

    aps = []
    for c in range(NUM_CLASSES):
        n_gt = class_n_gt[c]
        pc   = class_preds[c]
        if n_gt == 0 or not pc:
            continue
        pc.sort(key=lambda x: -x[0])
        tp_ = torch.tensor([p[1] for p in pc], dtype=torch.float32)
        fp_ = 1 - tp_
        cum_tp = tp_.cumsum(0); cum_fp = fp_.cumsum(0)
        rec  = cum_tp / (n_gt + 1e-6)
        prec = cum_tp / (cum_tp + cum_fp + 1e-6)
        aps.append(compute_ap(rec, prec))

    model.train()
    return float(np.mean(aps)) if aps else 0.0

print('mAP utility ready.')

# ── 9. Training Loop ──────────────────────────────────────────────────────────

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Mixed precision: Tensor Cores give a large throughput boost on conv/matmul
# in reduced precision. bfloat16 has the same exponent range as fp32, so
# gradients don't underflow and no GradScaler is needed (unlike float16).
# Only enabled when running on a CUDA device.
USE_AMP   = (DEVICE.type == 'cuda')
AMP_DTYPE = torch.bfloat16

# Linear warm-up for WARMUP_EPOCHS, then cosine decay to 0
WARMUP_EPOCHS = 5

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
    # Cosine decay to 10% of LR (not zero) so training stays alive late
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

history = {
    'train_loss': [], 'val_map': [],
    'loss_obj': [], 'loss_noobj': [], 'loss_coord': [], 'loss_cls': [],
}
best_map  = 0.0
best_ckpt = copy.deepcopy(model_raw.state_dict())  # always have something valid to save/load
EVAL_EVERY = 11       # evaluate mAP every N epochs
PATIENCE = 3          # in eval checkpoints, not epochs
patience_ctr = 0

print(f'Starting training: {NUM_EPOCHS} epochs on {DEVICE} | '
      f'batch_size={BATCH_SIZE} | AMP={"bf16" if USE_AMP else "off"} | '
      f'num_workers={NUM_WORKERS}')
if DEVICE.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    print(f'GPU: {torch.cuda.get_device_name(0)} | '
          f'total VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
t0 = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    sub = {'obj': 0.0, 'noobj': 0.0, 'coord': 0.0, 'cls': 0.0}

    for imgs, targets, ignore, _gts in tqdm(train_loader,
                               desc=f'Epoch {epoch}/{NUM_EPOCHS}', leave=False):
        imgs    = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        ignore  = ignore.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            preds = model(imgs)
            loss, comp = criterion(preds, targets, ignore)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # prevent exploding grads
        optimizer.step()

        epoch_loss += loss.item()
        for k in sub: sub[k] += comp[k]

    scheduler.step()
    nb = len(train_loader)
    history['train_loss'].append(epoch_loss / nb)
    for k in sub: history[f'loss_{k}'].append(sub[k] / nb)

    if epoch == 1 and DEVICE.type == 'cuda':
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'  Peak GPU memory after epoch 1: {peak_gb:.1f} / {total_gb:.1f} GB '
              f'({100*peak_gb/total_gb:.0f}%) — if this is well under ~80%, '
              f'BATCH_SIZE can likely go higher next run.')

    val_map = 0.0
    # Make sure to not miss a better checkpoint after initial training
    if epoch == 10:
        EVAL_EVERY = 5
    if epoch == 40:
        EVAL_EVERY = 4
    if epoch == 60:
        EVAL_EVERY = 2
    if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
        val_map = evaluate_map(model, val_loader)
        history['val_map'].append((epoch, val_map))
        if val_map > best_map:
            best_map  = val_map
            best_ckpt = copy.deepcopy(model_raw.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f'Early stopping at epoch {epoch}')
                break
        print(f'  Epoch {epoch:3d} | loss {epoch_loss/nb:.4f} | '
              f'mAP@0.50 {val_map*100:.2f}%  [best {best_map*100:.2f}%]  '
              f'lr {scheduler.get_last_lr()[0]:.2e}')
    else:
        print(f'  Epoch {epoch:3d} | loss {epoch_loss/nb:.4f}  '
              f'lr {scheduler.get_last_lr()[0]:.2e}')

print(f'\nTraining done in {(time.time()-t0)/60:.1f} min')
print(f'Best mAP@0.50 = {best_map*100:.2f}%')
torch.save(best_ckpt, os.path.join(OUTPUT_DIR, 'best_detector.pt'))
print(f"Saved best_detector.pt -> {os.path.join(OUTPUT_DIR, 'best_detector.pt')}")

# ── 10. Training Curves ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(history['train_loss'], color='#2563eb')
axes[0].set(xlabel='Epoch', ylabel='Loss', title='Total Training Loss')
axes[0].grid(alpha=0.3)

colors = {'loss_obj': '#16a34a', 'loss_noobj': '#dc2626',
          'loss_coord': '#d97706', 'loss_cls': '#7c3aed'}
for key, col in colors.items():
    axes[1].plot(history[key], label=key.replace('loss_', ''), color=col)
axes[1].set(xlabel='Epoch', ylabel='Loss', title='Loss Components')
axes[1].legend(); axes[1].grid(alpha=0.3)

if history['val_map']:
    ep_vals, map_vals = zip(*history['val_map'])
    axes[2].plot(ep_vals, [m*100 for m in map_vals], 'o-', color='#dc2626')
axes[2].set(xlabel='Epoch', ylabel='mAP (%)', title='Validation mAP@0.50')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved training_curves.png')

# ── 11. Detection Visualisation ───────────────────────────────────────────────

model_raw.load_state_dict(best_ckpt)
model.eval()

_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])

def to_rgb(tensor):
    """Denormalise (3, H, W) tensor → (H, W, 3) uint8 numpy array."""
    t = tensor.cpu() * _STD[:, None, None] + _MEAN[:, None, None]
    return (t.permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)

CMAP = plt.colormaps['tab20'].resampled(NUM_CLASSES)

def draw_detections(ax, img_np, boxes, scores, labels):
    ax.imshow(img_np)
    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = b.tolist()
        color = CMAP(int(l))
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, lw=2, edgecolor=color, facecolor='none'))
        ax.text(x1, max(y1-4, 0), f'{VOC_CLASSES[int(l)]} {s:.2f}',
                color='white', fontsize=7,
                bbox=dict(fc=color, alpha=0.75, pad=1, ec='none'))
    ax.axis('off')


sample_idx = random.sample(range(len(val_ds)), 8)
fig, axes  = plt.subplots(2, 4, figsize=(20, 10))

with torch.no_grad():
    for ax, idx in zip(axes.flatten(), sample_idx):
        img_t, _, _, _ = val_ds[idx]
        raw = model(img_t.unsqueeze(0).to(DEVICE))[0]
        boxes, scores, labels = decode_predictions(raw)
        boxes, scores, labels = nms_per_class(boxes, scores, labels)
        draw_detections(ax, to_rgb(img_t), boxes, scores, labels)

plt.suptitle('ScratchDetector — Validation Detections', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'detections.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved detections.png')

# ── Diagnostic: per-class box count + pairwise IoU spread for one image ──────
idx = sample_idx[3]
img_t, _, _, _ = val_ds[idx]
raw = model(img_t.unsqueeze(0).to(DEVICE))[0]
boxes, scores, labels = decode_predictions(raw)
boxes, scores, labels = nms_per_class(boxes, scores, labels)  # default CONF_THRESH=0.25

print(f"Surviving boxes: {boxes.shape[0]}")
for cls_id in labels.unique():
    m = labels == cls_id
    cls_boxes = boxes[m]
    if cls_boxes.shape[0] > 1:
        ious = torchvision.ops.box_iou(cls_boxes, cls_boxes)
        off_diag = ious[~torch.eye(len(ious), dtype=torch.bool)]
        print(f"class {int(cls_id)}: {cls_boxes.shape[0]} boxes | "
              f"mean pairwise IoU: {off_diag.mean():.3f} | max: {off_diag.max():.3f}")
        print("box widths:", (cls_boxes[:,2]-cls_boxes[:,0]).tolist())
        print("box heights:", (cls_boxes[:,3]-cls_boxes[:,1]).tolist())

# ── 12. Per-class AP Report ───────────────────────────────────────────────────
# Breaks down mAP@0.50 per VOC category for the report / presentation.

model_raw.load_state_dict(best_ckpt)
model.eval()

cls_preds_all = {c: [] for c in range(NUM_CLASSES)}
cls_ngt_all   = {c: 0  for c in range(NUM_CLASSES)}

with torch.no_grad():
    for imgs, targets, ignore, gts in tqdm(val_loader, desc='Per-class AP'):
        imgs    = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            preds = model(imgs)
        for i in range(imgs.shape[0]):
            boxes, scores, labels = nms_per_class(*decode_predictions(preds[i]),
                                                   conf_thresh=EVAL_CONF_THRESH)
            gt_single = {k: v.to(DEVICE) for k, v in gts[i].items()}
            gt_list, diff_list = _organize_gt(gt_single)
            for c, v in gt_list.items():
                cls_ngt_all[c] += int((~diff_list[c]).sum().item())
            matched = {c: [False]*len(v) for c, v in gt_list.items()}
            for k in range(len(boxes)):
                cls_id = int(labels[k])
                score  = float(scores[k])
                if cls_id not in gt_list or len(gt_list[cls_id]) == 0:
                    cls_preds_all[cls_id].append((score, 0)); continue
                gt_b = gt_list[cls_id]
                ious = box_iou_single(boxes[k], gt_b)
                best_iou, best_j = ious.max(0); best_j = best_j.item()
                if best_iou >= MAP_IOU_THRESH:
                    if diff_list[cls_id][best_j]:
                        continue  # matched a difficult GT: ignore detection entirely
                    if not matched[cls_id][best_j]:
                        matched[cls_id][best_j] = True
                        cls_preds_all[cls_id].append((score, 1))
                    else:
                        cls_preds_all[cls_id].append((score, 0))
                else:
                    cls_preds_all[cls_id].append((score, 0))

print(f'\n{"Class":<16} {"AP@50":>8}  {"GT boxes":>10}')
print('-' * 40)
aps_all = []
for c in range(NUM_CLASSES):
    n_gt = cls_ngt_all[c]
    pc   = cls_preds_all[c]
    if n_gt == 0 or not pc:
        ap = 0.0
    else:
        pc.sort(key=lambda x: -x[0])
        tp_ = torch.tensor([p[1] for p in pc], dtype=torch.float32)
        fp_ = 1 - tp_
        cum_tp = tp_.cumsum(0); cum_fp = fp_.cumsum(0)
        rec  = cum_tp / (n_gt + 1e-6)
        prec = cum_tp / (cum_tp + cum_fp + 1e-6)
        ap   = compute_ap(rec, prec)
    aps_all.append(ap)
    print(f'{VOC_CLASSES[c]:<16} {ap*100:>7.2f}%  {n_gt:>10}')

print('-' * 40)
print(f'{"mAP@0.50":<16} {np.mean(aps_all)*100:>7.2f}%')