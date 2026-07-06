# scratch-detector-voc

A YOLO-style object detector trained **entirely from scratch** (no pretrained weights) on a 5-class subset of Pascal VOC 2012 — custom ResNet-style backbone, anchor-based detection head, five-term composite loss, per-class NMS, and standard VOC mAP@0.50 evaluation.

Best result: **mAP@0.50 = 33.09%** on the validation set (aeroplane, bicycle, bus, car, motorbike).

## Pipeline

```
Dataset → Backbone → Detection Head → Loss → Decode + NMS → mAP Evaluation
```

- **Dataset**: Pascal VOC 2012, downloaded automatically via `torchvision`
- **Backbone**: 5-stage custom ResNet-style CNN (21.96M params, 416×416 → 13×13×512)
- **Head**: predicts objectness, box offsets, and class logits for 5 anchors per grid cell
- **Loss**: objectness (BCE) + no-object (BCE) + localization (Smooth L1) + classification (BCE + label smoothing) + background suppression (BCE)
- **Post-processing**: per-class confidence thresholding + NMS
- **Evaluation**: mAP@0.50, VOC 11-point interpolation

## Project layout

| File | Purpose |
|---|---|
| `config.py` | Hyperparameters and constants |
| `dataset.py` | VOC download, augmentation, dataset/dataloader construction |
| `model.py` | Backbone + detection head |
| `loss.py` | Five-term detection loss |
| `postprocess.py` | Box decoding + per-class NMS |
| `evaluate.py` | mAP@0.50 evaluation |
| `visualize.py` | Training curves + detection plots |
| `compute_anchors.py` | Standalone k-means anchor-box derivation |
| `train.py` | Entry point — trains the model and reports results |

## Usage

```bash
pip install -r requirements.txt
python train.py
```

Trains for 80 epochs and writes `best_detector.pt`, `training_curves.png`, and `detections.png` to `~/workspace/outputs`.
