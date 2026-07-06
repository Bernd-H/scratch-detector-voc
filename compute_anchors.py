"""Derives anchor box shapes via k-means clustering on VOC ground-truth boxes.

Standalone utility — run once to (re)generate the ANCHORS list in config.py.
Not part of the training pipeline itself.

Usage: python compute_anchors.py
"""
import numpy as np
from sklearn.cluster import KMeans

from config import DATA_ROOT, GRID_SIZE, SEED, VOC_CLASSES
from dataset import VOCDetectionDataset, download_voc


def collect_wh(dataset):
    whs = []
    for img_id, objs in dataset.samples:
        for o in objs:
            if o['name'] not in VOC_CLASSES:
                continue
            bb = o['bndbox']
            w = (float(bb['xmax']) - float(bb['xmin'])) / o['orig_w'] * GRID_SIZE
            h = (float(bb['ymax']) - float(bb['ymin'])) / o['orig_h'] * GRID_SIZE
            whs.append([w, h])
    return np.array(whs)


if __name__ == '__main__':
    download_voc()
    train_ds = VOCDetectionDataset(DATA_ROOT, image_set='train')
    wh = collect_wh(train_ds)
    km = KMeans(n_clusters=5, random_state=SEED).fit(wh)
    anchors = sorted(km.cluster_centers_.tolist(), key=lambda x: x[0] * x[1])
    print('ANCHORS (width, height) in grid-cell units:')
    for a in anchors:
        print(f'    ({a[0]:.2f}, {a[1]:.2f}),')
