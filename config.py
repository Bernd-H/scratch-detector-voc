"""Constants and hyperparameters shared across the training pipeline."""
import os
import random

import numpy as np
import torch

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

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.expanduser('~/workspace/outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

VOC_ROOT = os.path.expanduser('~/data')
os.makedirs(VOC_ROOT, exist_ok=True)
DATA_ROOT = os.path.join(VOC_ROOT, 'VOCdevkit', 'VOC2012')

# ── Model / grid geometry ─────────────────────────────────────────────────────
IMG_SIZE  = 416          # input resolution (square)
GRID_SIZE = 13           # feature-map grid = IMG_SIZE // 32

# Five anchors chosen by k-means clustering on VOC 2012 GT boxes
# (see compute_anchors.py). Format: (width, height) in grid-cell units
ANCHORS = [
    (1.08, 1.19),
    (3.42, 4.41),
    (6.63, 11.38),
    (9.42, 5.11),
    (16.62, 10.52),
]
NUM_ANCHORS = len(ANCHORS)
IGNORE_IOU_THRESH = 0.5

# ── Training schedule ─────────────────────────────────────────────────────────
# LR scaled with BATCH_SIZE (linear scaling rule) since larger batches need
# a proportionally larger LR to take equivalently-sized steps per epoch.
BATCH_SIZE   = 64
NUM_EPOCHS   = 80
LR           = 3e-4 * (BATCH_SIZE / 16)
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
NUM_WORKERS  = 8

# ── Loss weights (lambda values — YOLO-style) ────────────────────────────────
LAMBDA_OBJ    = 1.0
LAMBDA_NOOBJ  = 1.0
LAMBDA_COORD  = 5.0
LAMBDA_CLS    = 1.0
LAMBDA_CLS_BG = 0.5   # background class-suppression weight

# Label smoothing applied inside Dataset targets
LABEL_SMOOTHING = 0.1

# ── Inference thresholds ──────────────────────────────────────────────────────
CONF_THRESH    = 0.45
NMS_IOU_THRESH = 0.35

# Separate, much lower threshold for mAP evaluation: AP integrates precision
# over the FULL recall range, so eval must keep low-confidence boxes too.
# CONF_THRESH is only for the human-facing visualisation.
EVAL_CONF_THRESH = 0.001

# mAP evaluation IoU threshold
MAP_IOU_THRESH = 0.50

# ── Dataset classes ───────────────────────────────────────────────────────────
VOC_CLASSES = [
    "aeroplane", "bicycle",
    "bus", "car", "motorbike",
]
NUM_CLASSES = len(VOC_CLASSES)

# Ground-truth instance counts for this 5-class subset, used to derive
# inverse-frequency class weights for the classification loss and sampler.
CLASS_INSTANCE_COUNTS = {
    'car': 1191, 'aeroplane': 470,
    'bicycle': 410, 'motorbike': 375, 'bus': 317,
}

# ── Mixed precision ───────────────────────────────────────────────────────────
# Tensor Cores give a large throughput boost on conv/matmul in reduced
# precision. bfloat16 has the same exponent range as fp32, so gradients
# don't underflow and no GradScaler is needed (unlike float16).
USE_AMP   = (DEVICE.type == 'cuda')
AMP_DTYPE = torch.bfloat16
