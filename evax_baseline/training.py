"""
training.py
===========
Train / validate the multi-task model.

Loss:   L = L_seg(DiceCE) + lambda_cls * BCE(cls)
Optim:  SGD, momentum 0.99, nesterov, weight_decay 3e-5  (nnU-Net defaults)
LR:     poly decay (nnU-Net)
AMP:    autocast + GradScaler on cuda
Logs:   train loss, val seg-dice (mean over fg), val cls accuracy
Saves:  best checkpoint by mean validation dice
"""

import os
import numpy as np
import torch
from torch import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import DiceCELoss, dice_metric, poly_lr, save_ckpt


def save_history_plot(history, out_path):
    """Save train/val loss + val dice + val cls_acc curves to a png."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # left: losses
    ax[0].plot(epochs, history['train_loss'], label='train loss')
    ax[0].plot(epochs, history['val_loss'], label='val loss')
    ax[0].set_xlabel('epoch'); ax[0].set_ylabel('loss')
    ax[0].set_title('Loss'); ax[0].legend(); ax[0].grid(alpha=0.3)

    # right: val dice + cls acc
    ax[1].plot(epochs, history['val_dice'], label='val dice', color='tab:green')
    ax[1].plot(epochs, history['val_cls_acc'], label='val cls acc', color='tab:orange')
    ax[1].set_xlabel('epoch'); ax[1].set_ylabel('metric')
    ax[1].set_title('Validation Dice / Cls Acc')
    ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=90)
    plt.close()

import math

def compute_lr(epoch, max_epochs, initial_lr, scheduler='poly', warmup_epochs=0):
    # warmup: 0 -> initial_lr 선형 증가
    if warmup_epochs and epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    if scheduler == 'cosine':
        prog = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * initial_lr * (1 + math.cos(math.pi * prog))
    # poly (nnU-Net 기본) — 기존 동작 유지
    from utils import poly_lr
    return poly_lr(epoch, max_epochs, initial_lr)

def make_optimizer(model, initial_lr=1e-2, weight_decay=3e-5,
                   optimizer_name='sgd'):
    if optimizer_name == 'adamw':
        # ViT fine-tuning: AdamW, norm/bias에는 weight decay 미적용
        wd = 0.05
        if hasattr(model, 'param_groups'):
            groups = model.param_groups(initial_lr, backbone_lr_mult=1.0,
                                        weight_decay=wd)
        else:
            decay, no_decay = [], []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim <= 1 or n.endswith('.bias') else decay).append(p)
            groups = [{'params': decay, 'weight_decay': wd},
                      {'params': no_decay, 'weight_decay': 0.0}]
        return torch.optim.AdamW(groups, lr=initial_lr, betas=(0.9, 0.999))

    return torch.optim.SGD(model.parameters(), lr=initial_lr,
                           momentum=0.99, nesterov=True,
                           weight_decay=weight_decay)


def _seg_loss_on_output(seg_output, target, seg_loss_fn):
    """Handle deep-supervision (list) or single tensor."""
    if isinstance(seg_output, (list, tuple)):
        # simple equal-ish weighting if DS is on; highest res gets most weight
        weights = [1 / (2 ** i) for i in range(len(seg_output))]
        weights[-1] = 0.0
        s = sum(weights)
        weights = [w / s for w in weights]
        loss = 0.0
        for w, o in zip(weights, seg_output):
            if w == 0:
                continue
            # target may need downsampling for DS; here we assume DS off by default
            loss = loss + w * seg_loss_fn(o, target)
        return loss
    return seg_loss_fn(seg_output, target)


def train_one_epoch(model, loader, optimizer, seg_loss_fn, lambda_cls,
                    device, scaler, use_amp=True):
    model.train()
    losses = []
    bce = torch.nn.BCEWithLogitsLoss()
    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        cls = batch['cls'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device.type, enabled=use_amp) if device.type == 'cuda' else _nullctx():
            model.return_cls = True
            seg_out, cls_logit = model(img)
            l_seg = _seg_loss_on_output(seg_out, mask, seg_loss_fn)
            l_cls = bce(cls_logit, cls)
            loss = l_seg + lambda_cls * l_cls

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            optimizer.step()

        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def validate(model, loader, seg_loss_fn, lambda_cls, device, use_amp=True):
    model.eval()
    dices, cls_correct, cls_total = [], 0, 0
    bce = torch.nn.BCEWithLogitsLoss()
    val_losses = []
    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        cls = batch['cls'].to(device, non_blocking=True)

        with autocast(device.type, enabled=use_amp) if device.type == 'cuda' else _nullctx():
            model.return_cls = True
            seg_out, cls_logit = model(img)
            seg_main = seg_out[0] if isinstance(seg_out, (list, tuple)) else seg_out
            l_seg = seg_loss_fn(seg_main, mask)
            l_cls = bce(cls_logit, cls)
            loss = l_seg + lambda_cls * l_cls
        val_losses.append(loss.item())

        # per-sample dice
        pred = seg_main.argmax(1).cpu().numpy()        # (B, H, W)
        gt = mask.squeeze(1).cpu().numpy()
        for b in range(pred.shape[0]):
            d = dice_metric(pred[b], gt[b])
            dices.append(d)

        # cls accuracy
        cls_pred = (torch.sigmoid(cls_logit) > 0.5).float()
        cls_correct += (cls_pred == cls).sum().item()
        cls_total += cls.numel()

    mean_dice = float(np.nanmean(dices)) if len(dices) else 0.0
    cls_acc = cls_correct / max(cls_total, 1)
    return float(np.mean(val_losses)), mean_dice, cls_acc


#def fit(model, train_loader, val_loader, device,
#        max_epochs=1000, lambda_cls=0.1,
#        initial_lr=1e-2, ckpt_path="best.pth",
#        patience=None, batch_dice=True, optimizer_name='sgd'):

def fit(model, train_loader, val_loader, device,
        max_epochs=1000, lambda_cls=0.1,
        initial_lr=1e-2, ckpt_path="best.pth",
        patience=None, batch_dice=True, optimizer_name='sgd',
        scheduler='poly', warmup_epochs=0, loss_name='dicece'):

    optimizer = make_optimizer(model, initial_lr=initial_lr,
                               optimizer_name=optimizer_name)
    if loss_name == 'tversky':
        from utils import TverskyCELoss
        seg_loss_fn = TverskyCELoss(alpha=0.3, beta=0.7, gamma=1.3333,
                                    batch_dice=batch_dice)
    elif loss_name == 'boundary':
        from utils import BoundaryDiceLoss
        seg_loss_fn = BoundaryDiceLoss(batch_dice=batch_dice, w_boundary=0.5)
    else:
        seg_loss_fn = DiceCELoss(batch_dice=batch_dice)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_cls_acc': []}
    plot_path = os.path.splitext(ckpt_path)[0] + "_curves.png"

    best_score, best_epoch, since_improve = -1.0, -1, 0
    for epoch in range(max_epochs):
        #for g in optimizer.param_groups:
        #    g['lr'] = poly_lr(epoch, max_epochs, initial_lr)
        for g in optimizer.param_groups:
            g['lr'] = compute_lr(epoch, max_epochs, initial_lr,
                                 scheduler, warmup_epochs)        

        tr_loss = train_one_epoch(model, train_loader, optimizer, seg_loss_fn,
                                  lambda_cls, device, scaler)
        val_loss, val_dice, val_acc = validate(model, val_loader, seg_loss_fn,
                                               lambda_cls, device)

        # record history + refresh plot every epoch
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_cls_acc'].append(val_acc)
        save_history_plot(history, plot_path)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"[{epoch:04d}] lr={lr_now:.5f} "
              f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
              f"dice={val_dice:.4f} cls_acc={val_acc:.4f}")

        if lambda_cls > 0:
            final_score = 0.7 * val_acc + 0.3 * val_dice
        else:
            final_score = val_dice          # cls 미학습 -> dice로만 선택
        improved = final_score > best_score
        improved = final_score > best_score
        if improved:
            best_score, best_epoch, since_improve = final_score, epoch, 0
            save_ckpt(ckpt_path, model, optimizer, epoch, final_score)
        
        else:
            since_improve += 1

        if patience is not None and since_improve >= patience:
            print(f"Early stop at epoch {epoch} "
                  f"(best final {best_score:.4f} @ {best_epoch})")
            break

    print(f"Done. best final {best_score:.4f} @ epoch {best_epoch}")
    print(f"curves saved to {plot_path}")
    return best_score, best_epoch


# small null-context for non-cuda
import contextlib
@contextlib.contextmanager
def _nullctx():
    yield