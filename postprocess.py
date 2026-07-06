"""Decoding raw model output to pixel-space boxes, and per-class NMS."""
import torch
import torchvision

from config import ANCHORS, CONF_THRESH, GRID_SIZE, IMG_SIZE, NMS_IOU_THRESH

CELL_SIZE    = IMG_SIZE / GRID_SIZE                        # pixels per grid cell
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
