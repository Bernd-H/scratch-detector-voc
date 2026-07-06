"""Pascal VOC dataset loading, augmentation, and YOLO-style target encoding."""
import math
import os
import random
import xml.etree.ElementTree as ET

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets

from config import (
    ANCHORS, BATCH_SIZE, CLASS_INSTANCE_COUNTS, DATA_ROOT, DEVICE, GRID_SIZE,
    IGNORE_IOU_THRESH, IMG_SIZE, LABEL_SMOOTHING, NUM_ANCHORS, NUM_CLASSES, NUM_WORKERS,
    VOC_CLASSES, VOC_ROOT,
)

_anchors_t = torch.tensor(ANCHORS, dtype=torch.float32)

train_transform = T.Compose([
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def download_voc():
    print('Downloading Pascal VOC 2012 (skipped automatically if already present)...')
    datasets.VOCDetection(root=VOC_ROOT, year='2012', image_set='train', download=True)
    print('VOC 2012 ready.')


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


def class_presence_weight(objects, class_freq_inv):
    present = [c for c in VOC_CLASSES if any(o['name'] == c for o in objects)]
    if not present:
        return 1.0
    return max(class_freq_inv[c] for c in present)


def compute_class_weights():
    """Inverse-frequency class weights (sqrt-damped) for the classification loss."""
    counts_tensor = torch.tensor(
        [CLASS_INSTANCE_COUNTS[c] for c in VOC_CLASSES], dtype=torch.float32
    )
    raw_weights = counts_tensor.sum() / (NUM_CLASSES * counts_tensor)
    return raw_weights.sqrt().to(DEVICE)


def detection_collate(batch):
    """Custom collate: imgs/target/ignore stack normally (fixed shape per
    sample); gt dicts have variable-length boxes/labels/difficult, so they
    stay as a plain list of dicts, one per sample in the batch."""
    imgs, targets, ignores, gts = zip(*batch)
    imgs    = torch.stack(imgs, dim=0)
    targets = torch.stack(targets, dim=0)
    ignores = torch.stack(ignores, dim=0)
    return imgs, targets, ignores, list(gts)


def build_dataloaders(batch_size=BATCH_SIZE):
    """Downloads VOC if needed and builds train/val datasets + dataloaders."""
    download_voc()

    train_ds = VOCDetectionDataset(DATA_ROOT, image_set='train', transform=train_transform)
    val_ds   = VOCDetectionDataset(DATA_ROOT, image_set='val',   transform=val_transform)

    class_freq_inv = {c: 1.0 / CLASS_INSTANCE_COUNTS[c] for c in VOC_CLASSES}
    sample_weights = [
        class_presence_weight(objs, class_freq_inv) for _, objs in train_ds.samples
    ]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=detection_collate,
                              persistent_workers=True, prefetch_factor=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=detection_collate,
                              persistent_workers=True, prefetch_factor=4)

    print(f'Train: {len(train_ds):,} images   Val: {len(val_ds):,} images')
    return train_ds, val_ds, train_loader, val_loader
