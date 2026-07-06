"""Training-curve and detection-box plotting utilities."""
import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import OUTPUT_DIR

_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


def to_rgb(tensor):
    """Denormalise (3, H, W) tensor → (H, W, 3) uint8 numpy array."""
    t = tensor.cpu() * _STD[:, None, None] + _MEAN[:, None, None]
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def draw_detections(ax, img_np, boxes, scores, labels, class_names, cmap):
    ax.imshow(img_np)
    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = b.tolist()
        color = cmap(int(l))
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, lw=2, edgecolor=color, facecolor='none'))
        ax.text(x1, max(y1 - 4, 0), f'{class_names[int(l)]} {s:.2f}',
                color='white', fontsize=7,
                bbox=dict(fc=color, alpha=0.75, pad=1, ec='none'))
    ax.axis('off')


def plot_training_curves(history):
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
        axes[2].plot(ep_vals, [m * 100 for m in map_vals], 'o-', color='#dc2626')
    axes[2].set(xlabel='Epoch', ylabel='mAP (%)', title='Validation mAP@0.50')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved {out_path}')


def plot_sample_detections(model, val_ds, device, class_names, n=8):
    """Runs inference on `n` random validation images and saves a detection grid."""
    from postprocess import decode_predictions, nms_per_class

    cmap = plt.colormaps['tab20'].resampled(len(class_names))
    sample_idx = random.sample(range(len(val_ds)), n)
    fig, axes = plt.subplots(2, n // 2, figsize=(5 * (n // 2), 10))

    model.eval()
    with torch.no_grad():
        for ax, idx in zip(axes.flatten(), sample_idx):
            img_t, _, _, _ = val_ds[idx]
            raw = model(img_t.unsqueeze(0).to(device))[0]
            boxes, scores, labels = decode_predictions(raw)
            boxes, scores, labels = nms_per_class(boxes, scores, labels)
            draw_detections(ax, to_rgb(img_t), boxes, scores, labels, class_names, cmap)

    plt.suptitle('Validation Detections', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'detections.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved {out_path}')
