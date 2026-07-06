"""Trains the from-scratch YOLO-style detector on Pascal VOC and reports mAP@0.50.

Usage: python train.py
"""
import copy
import math
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from config import (
    AMP_DTYPE, BATCH_SIZE, DEVICE, GRID_SIZE, IMG_SIZE, LR, NUM_ANCHORS, NUM_CLASSES,
    NUM_EPOCHS, NUM_WORKERS, OUTPUT_DIR, USE_AMP, VOC_CLASSES, WARMUP_EPOCHS, WEIGHT_DECAY,
)
from dataset import build_dataloaders, compute_class_weights
from evaluate import evaluate_map, per_class_ap_report
from loss import DetectionLoss
from model import ScratchDetector
from visualize import plot_sample_detections, plot_training_curves

EVAL_EVERY = 11   # evaluate mAP every N epochs (ramps up later in training)
PATIENCE   = 3    # early-stopping patience, in eval checkpoints (not epochs)


def build_model():
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
    return model, model_raw


def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
    # Cosine decay to 10% of LR (not zero) so training stays alive late
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def train():
    print('Device:', DEVICE)

    train_ds, val_ds, train_loader, val_loader = build_dataloaders(BATCH_SIZE)
    class_weights = compute_class_weights()

    model, model_raw = build_model()
    criterion = DetectionLoss(class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {
        'train_loss': [], 'val_map': [],
        'loss_obj': [], 'loss_noobj': [], 'loss_coord': [], 'loss_cls': [],
    }
    best_map  = 0.0
    best_ckpt = copy.deepcopy(model_raw.state_dict())
    eval_every = EVAL_EVERY
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
            for k in sub:
                sub[k] += comp[k]

        scheduler.step()
        nb = len(train_loader)
        history['train_loss'].append(epoch_loss / nb)
        for k in sub:
            history[f'loss_{k}'].append(sub[k] / nb)

        if epoch == 1 and DEVICE.type == 'cuda':
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f'  Peak GPU memory after epoch 1: {peak_gb:.1f} / {total_gb:.1f} GB '
                  f'({100*peak_gb/total_gb:.0f}%)')

        # Evaluate more frequently as training progresses, to not miss the
        # best checkpoint once the model is close to convergence.
        if epoch == 10:
            eval_every = 5
        if epoch == 40:
            eval_every = 4
        if epoch == 60:
            eval_every = 2

        if epoch % eval_every == 0 or epoch == NUM_EPOCHS:
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
    ckpt_path = os.path.join(OUTPUT_DIR, 'best_detector.pt')
    torch.save(best_ckpt, ckpt_path)
    print(f'Saved {ckpt_path}')

    plot_training_curves(history)

    model_raw.load_state_dict(best_ckpt)
    plot_sample_detections(model, val_ds, DEVICE, VOC_CLASSES)

    per_class_ap_report(model, val_loader, VOC_CLASSES)


if __name__ == '__main__':
    train()
