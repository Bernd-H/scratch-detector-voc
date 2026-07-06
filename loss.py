"""Five-term YOLO-style detection loss.

    L = λ_obj    * BCE(σ(logit), 1)           [cells WITH an object]
      + λ_noobj  * BCE(σ(logit), 0)           [cells WITHOUT an object]
      + λ_coord  * SmoothL1(pred_box, gt_box) [cells WITH an object]
      + λ_cls    * BCE(pred_cls, gt_cls_soft) [cells WITH an object]
      + λ_cls_bg * BCE(pred_cls, 0)           [cells WITHOUT an object]

Class targets already carry label-smoothing from the Dataset, so we use
BCEWithLogitsLoss (not CrossEntropyLoss) for the class term.
"""
import torch
import torch.nn as nn

from config import LAMBDA_CLS, LAMBDA_CLS_BG, LAMBDA_COORD, LAMBDA_NOOBJ, LAMBDA_OBJ


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
