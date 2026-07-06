"""Standard VOC 11-point interpolated mAP@0.50 evaluation."""
import numpy as np
import torch
from tqdm import tqdm

from config import AMP_DTYPE, DEVICE, EVAL_CONF_THRESH, MAP_IOU_THRESH, NUM_CLASSES, USE_AMP
from postprocess import decode_predictions, nms_per_class


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


def _aggregate_predictions(model, loader, conf_thresh, max_batches=None, desc='mAP eval'):
    """Runs the model over a loader and collects (score, is_tp) per class,
    plus the per-class GT count, following the VOC evaluation protocol
    (Everingham et al.): GT boxes flagged `difficult` are excluded from the
    GT count (they don't contribute to recall), and any detection whose best
    match is a difficult GT box is simply discarded — neither a true positive
    nor a false positive — rather than penalised.
    """
    class_preds = {c: [] for c in range(NUM_CLASSES)}
    class_n_gt  = {c: 0  for c in range(NUM_CLASSES)}

    for batch_idx, (imgs, targets, ignore, gts) in enumerate(
            tqdm(loader, desc=desc, leave=False)):
        if max_batches and batch_idx >= max_batches:
            break
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            preds = model(imgs)

        for i in range(imgs.shape[0]):
            boxes, scores, labels = decode_predictions(preds[i])
            boxes, scores, labels = nms_per_class(boxes, scores, labels,
                                                   conf_thresh=conf_thresh)

            gt_single = {k: v.to(DEVICE) for k, v in gts[i].items()}
            gt_list, diff_list = _organize_gt(gt_single)
            for cls_ in gt_list:
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
                if best_iou >= MAP_IOU_THRESH:
                    if diff_list[cls_id][best_j]:
                        continue  # matched a difficult GT: ignore detection entirely
                    if not matched[cls_id][best_j]:
                        matched[cls_id][best_j] = True
                        class_preds[cls_id].append((score, 1))
                    else:
                        class_preds[cls_id].append((score, 0))
                else:
                    class_preds[cls_id].append((score, 0))

    return class_preds, class_n_gt


def _aps_from_predictions(class_preds, class_n_gt, skip_empty=False):
    """Computes per-class AP. If skip_empty, classes with no GT or no
    predictions are omitted entirely (for mAP averaging); otherwise they're
    reported as AP=0.0 (for a complete per-class table)."""
    aps = []
    for c in range(NUM_CLASSES):
        n_gt = class_n_gt[c]
        pc   = class_preds[c]
        if n_gt == 0 or not pc:
            if not skip_empty:
                aps.append(0.0)
            continue
        pc.sort(key=lambda x: -x[0])
        tp_ = torch.tensor([p[1] for p in pc], dtype=torch.float32)
        fp_ = 1 - tp_
        cum_tp = tp_.cumsum(0); cum_fp = fp_.cumsum(0)
        rec  = cum_tp / (n_gt + 1e-6)
        prec = cum_tp / (cum_tp + cum_fp + 1e-6)
        aps.append(compute_ap(rec, prec))
    return aps


@torch.no_grad()
def evaluate_map(model, loader, max_batches=None):
    """Compute mAP@MAP_IOU_THRESH over a DataLoader."""
    model.eval()
    class_preds, class_n_gt = _aggregate_predictions(
        model, loader, conf_thresh=EVAL_CONF_THRESH, max_batches=max_batches)
    aps = _aps_from_predictions(class_preds, class_n_gt, skip_empty=True)
    model.train()
    return float(np.mean(aps)) if aps else 0.0


@torch.no_grad()
def per_class_ap_report(model, loader, class_names):
    """Prints a per-class AP@0.50 breakdown, for the report/presentation."""
    model.eval()
    class_preds, class_n_gt = _aggregate_predictions(
        model, loader, conf_thresh=EVAL_CONF_THRESH, desc='Per-class AP')
    aps = _aps_from_predictions(class_preds, class_n_gt)
    model.train()

    print(f'\n{"Class":<16} {"AP@50":>8}  {"GT boxes":>10}')
    print('-' * 40)
    for c, ap in enumerate(aps):
        print(f'{class_names[c]:<16} {ap*100:>7.2f}%  {class_n_gt[c]:>10}')
    print('-' * 40)
    print(f'{"mAP@0.50":<16} {np.mean(aps)*100:>7.2f}%')
    return aps
