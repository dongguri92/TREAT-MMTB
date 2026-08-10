"""
utils.py
========
- Dice + CE segmentation loss (matches nnU-Net's DC_and_CE, weight_dice=weight_ce=1)
- soft dice loss with batch_dice option
- dice metric for validation
- postprocessing: remove small components + suppress FP using the cls head
  (cls=absent -> empty the mask). This is the asymmetric rule we agreed on.
- checkpoint save/load
"""

from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ===========================================================================
#  Segmentation loss: Dice + CrossEntropy
# ===========================================================================
class SoftDiceLoss(nn.Module):
    """Soft Dice over foreground. batch_dice=True pools the batch (nnU-Net style),
    which is robust when many images in a batch are empty (negatives)."""
    def __init__(self, batch_dice=True, smooth=1e-5):
        super().__init__()
        self.batch_dice = batch_dice
        self.smooth = smooth

    def forward(self, logits, target):
        # logits: (B, 2, H, W); target: (B, 1, H, W) long
        probs = F.softmax(logits, dim=1)[:, 1]          # foreground prob (B,H,W)
        tgt = (target.squeeze(1) > 0).float()           # (B, H, W)

        if self.batch_dice:
            dims = (0, 1, 2)
        else:
            dims = (1, 2)

        inter = (probs * tgt).sum(dims)
        denom = probs.sum(dims) + tgt.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self, batch_dice=True, weight_dice=1.0, weight_ce=1.0):
        super().__init__()
        self.dice = SoftDiceLoss(batch_dice=batch_dice)
        self.ce = nn.CrossEntropyLoss()
        self.wd, self.wc = weight_dice, weight_ce

    def forward(self, logits, target):
        # CE expects target (B, H, W) long
        ce = self.ce(logits, target.squeeze(1).long())
        dc = self.dice(logits, target)
        return self.wd * dc + self.wc * ce

# =============================================================================
#  utils.py에 추가할 loss들 (기존 SoftDiceLoss / DiceCELoss 아래에 붙여넣기)
#  SoftDiceLoss의 softmax fg-prob 방식과 동일하게 맞춤.
# =============================================================================

class TverskyLoss(nn.Module):
    """Focal Tversky. under-segmentation(좁게 그림)을 겨냥해 FN 페널티를 키움.
       alpha=FP 가중, beta=FN 가중. beta>alpha 이면 더 넓게 그리도록 유도.
       gamma>1 이면 어려운(작은) 케이스에 focal 집중. gamma=1 이면 순수 Tversky."""
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.3333,
                 batch_dice=True, smooth=1e-5):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.batch_dice = batch_dice
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)[:, 1]          # (B,H,W)
        tgt = (target.squeeze(1) > 0).float()
        dims = (0, 1, 2) if self.batch_dice else (1, 2)

        tp = (probs * tgt).sum(dims)
        fp = (probs * (1 - tgt)).sum(dims)
        fn = ((1 - probs) * tgt).sum(dims)
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)
        # focal: 어려운 케이스 강조. gamma=1 이면 (1-tversky) 와 동일
        loss = (1 - tversky) ** self.gamma
        return loss.mean()


class TverskyCELoss(nn.Module):
    """Focal Tversky + CE. DiceCELoss 자리에 그대로 대체 가능."""
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.3333,
                 batch_dice=True, weight_tv=1.0, weight_ce=1.0):
        super().__init__()
        self.tv = TverskyLoss(alpha=alpha, beta=beta, gamma=gamma,
                              batch_dice=batch_dice)
        self.ce = nn.CrossEntropyLoss()
        self.wt, self.wc = weight_tv, weight_ce

    def forward(self, logits, target):
        ce = self.ce(logits, target.squeeze(1).long())
        tv = self.tv(logits, target)
        return self.wt * tv + self.wc * ce


def _dist_transform(mask_bool):
    """각 배경 픽셀의 전경까지 거리 - 전경 픽셀의 배경까지 거리 (level set phi).
       boundary loss 표준 정의. mask_bool: (H,W) numpy bool."""
    from scipy.ndimage import distance_transform_edt as edt
    posmask = mask_bool
    negmask = ~posmask
    if posmask.any():
        negative_distance = cast(Any, edt(negmask))
        positive_distance = cast(Any, edt(posmask))
        phi = negative_distance - positive_distance   # 밖은 +, 안은 -
    else:
        phi = edt(negmask)                  # 전경 없으면 전부 +
    return phi


class BoundaryDiceLoss(nn.Module):
    """Dice + w_b * Boundary. Boundary loss = mean(softmax_fg * phi),
       phi는 GT의 signed distance map. under-segment된 경계를 밀어냄.
       순수 boundary는 초반 불안정 -> Dice와 고정 가중 합으로 안정화."""
    def __init__(self, batch_dice=True, w_boundary=0.5, smooth=1e-5):
        super().__init__()
        self.dice = SoftDiceLoss(batch_dice=batch_dice)
        self.w_b = w_boundary

    def forward(self, logits, target):
        import numpy as np
        import torch
        dc = self.dice(logits, target)

        probs = F.softmax(logits, dim=1)[:, 1]          # (B,H,W)
        tgt = target.squeeze(1)                         # (B,H,W)
        # phi 계산은 numpy(CPU)에서 배치별로
        phis = np.empty(tgt.shape, dtype=np.float32)
        tgt_np = (tgt.detach().cpu().numpy() > 0)
        for b in range(tgt_np.shape[0]):
            phis[b] = _dist_transform(tgt_np[b])
        phi = torch.from_numpy(phis).to(probs.device)
        # 거리 스케일이 크므로 정규화(픽셀수 기준). 안 하면 loss가 폭주.
        phi = phi / (phi.abs().amax(dim=(1, 2), keepdim=True) + 1e-5)
        boundary = (probs * phi).mean()
        return dc + self.w_b * boundary


# ===========================================================================
#  Dice metric (hard), matches the challenge dice semantics for a single case
# ===========================================================================
@torch.no_grad()
def dice_metric(pred_mask, gt_mask):
    """pred_mask, gt_mask: (H, W) binary numpy or tensor.
    Returns dice in [0,1]; mirrors challenge edge cases:
      both empty -> nan (excluded), one empty -> 0."""
    p = (np.asarray(pred_mask) > 0)
    g = (np.asarray(gt_mask) > 0)
    ps, gs = p.sum(), g.sum()
    if ps == 0 and gs == 0:
        return np.nan
    if ps == 0 or gs == 0:
        return 0.0
    inter = np.logical_and(p, g).sum()
    return 2.0 * inter / (ps + gs)


# ===========================================================================
#  Postprocessing
# ===========================================================================
def remove_small_components(mask, min_pixels=50):
    """Remove connected components smaller than min_pixels.
    mask: (H, W) binary numpy. Uses scipy if available, else returns as-is."""
    try:
        from scipy import ndimage
    except ImportError:
        return mask
    lbl, n = cast(tuple[Any, int], ndimage.label(mask > 0))
    if n == 0:
        return mask
    out = np.zeros_like(mask)
    for i in range(1, n + 1):
        comp = (lbl == i)
        if comp.sum() >= min_pixels:
            out[comp] = 1
    return out


def postprocess(seg_mask, cls_prob, cls_threshold=0.5, min_pixels=50):
    """
    Apply the agreed asymmetric rule:
      1. remove small components from seg_mask
      2. if cls says ABSENT (cls_prob < threshold) -> empty the mask (FP suppress)
         if cls says PRESENT -> keep seg as-is (do NOT invent a mask)
    Returns (final_mask, detection_flag).
    detection: follow cls for the CSV (cls is the presence authority).
    """
    m = remove_small_components(seg_mask, min_pixels=min_pixels)
    present = cls_prob >= cls_threshold
    if not present:
        m = np.zeros_like(m)        # suppress false positive
    detection = int(present)
    return m, detection


# ===========================================================================
#  Checkpoint
# ===========================================================================
def save_ckpt(path, model, optimizer, epoch, best_metric):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric,
    }, path)


def load_ckpt(path, model, optimizer=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0), ckpt.get('best_metric', None)


# ===========================================================================
#  Poly LR (nnU-Net schedule)
# ===========================================================================
def poly_lr(epoch, max_epochs, initial_lr=1e-2, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent
