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
import random
import time

import matplotlib
import numpy as np
import torch
from torch import autocast
from torch.amp.grad_scaler import GradScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import DiceCELoss, dice_metric, save_ckpt


def save_history_plot(history, out_path):
    """Save train/val loss + val dice + val cls_acc curves to a png."""
    epochs = range(1, len(history['train_loss']) + 1)
    _fig, ax = plt.subplots(1, 2, figsize=(12, 5))

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


def _seg_loss_on_output(seg_output, target, seg_loss_fn) -> torch.Tensor:
    """Handle deep-supervision (list) or single tensor."""
    if isinstance(seg_output, (list, tuple)):
        # simple equal-ish weighting if DS is on; highest res gets most weight
        weights = [1 / (2 ** i) for i in range(len(seg_output))]
        weights[-1] = 0.0
        s = sum(weights)
        weights = [w / s for w in weights]
        loss = torch.zeros((), device=target.device)
        for w, o in zip(weights, seg_output):
            if w == 0:
                continue
            # target may need downsampling for DS; here we assume DS off by default
            loss = loss + w * seg_loss_fn(o, target)
        return loss
    return seg_loss_fn(seg_output, target)


def train_one_epoch(model, loader, optimizer, seg_loss_fn, lambda_cls,
                    device, scaler, use_amp=True, wandb_run=None,
                    epoch=0, global_step=0, gradient_accumulation_steps=1):
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    usable_micro_steps = (
        len(loader) // gradient_accumulation_steps
    ) * gradient_accumulation_steps
    if usable_micro_steps == 0:
        raise ValueError("loader cannot provide one complete effective batch")
    model.train()
    losses = []
    bce = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        if batch_index >= usable_micro_steps:
            break
        img = batch['image'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        cls = batch['cls'].to(device, non_blocking=True)

        with autocast(device.type, enabled=use_amp) if device.type == 'cuda' else _nullctx():
            model.return_cls = True
            seg_out, cls_logit = model(img)
            l_seg = _seg_loss_on_output(seg_out, mask, seg_loss_fn)
            l_cls = bce(cls_logit, cls)
            loss = l_seg + lambda_cls * l_cls
            backward_loss = loss / gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()

        optimizer_step_completed = (
            (batch_index + 1) % gradient_accumulation_steps == 0
        )
        if optimizer_step_completed:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        losses.append(loss.item())
        if wandb_run is not None:
            step_log = {
                'train/global_step': global_step,
                'train/micro_step': epoch * usable_micro_steps + batch_index + 1,
                'train/epoch': epoch + 1,
                'train/batch_loss': loss.item(),
                'train/total_loss': loss.item(),
                'train/segmentation_loss': l_seg.item(),
                'train/classification_loss': l_cls.item(),
                'train/optimizer_step_completed': optimizer_step_completed,
                'train/gradient_accumulation_steps': gradient_accumulation_steps,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
            }
            step_log.update({
                f'train/learning_rate_group_{index}': group['lr']
                for index, group in enumerate(optimizer.param_groups)
            })
            wandb_run.log(step_log)
    return float(np.mean(losses)), global_step


@torch.no_grad()
def validate(model, loader, seg_loss_fn, lambda_cls, device, use_amp=True,
             wandb_run=None, epoch=0, native_combo_expected_ids=None):
    model.eval()
    dices, cls_correct, cls_total = [], 0, 0
    native_cases = []
    bce = torch.nn.BCEWithLogitsLoss()
    val_losses = []
    for batch_index, batch in enumerate(loader):
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
        if wandb_run is not None:
            wandb_run.log({
                'validation/global_step': epoch * len(loader) + batch_index + 1,
                'validation/epoch': epoch + 1,
                'validation/batch_loss': loss.item(),
            })

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

        if native_combo_expected_ids is not None:
            from reproduction import native_case_record

            required = {
                'id', 'native_mask', 'native_shape', 'crop_shape', 'pad_info'
            }
            missing_fields = sorted(required - set(batch))
            if missing_fields:
                raise ValueError(
                    f"native validation metadata missing: {missing_fields}"
                )
            fg_prob = torch.softmax(seg_main, dim=1)[:, 1].cpu().numpy()
            cls_prob = torch.sigmoid(cls_logit).flatten().cpu().numpy()
            for index, case_id in enumerate(batch['id']):
                native_cases.append(native_case_record(
                    case_id=case_id,
                    foreground_probability=fg_prob[index],
                    cls_probability=cls_prob[index],
                    native_mask=batch['native_mask'][index].cpu().numpy(),
                    pad_info=batch['pad_info'][index].cpu().numpy(),
                    crop_shape=batch['crop_shape'][index].cpu().numpy(),
                    native_shape=batch['native_shape'][index].cpu().numpy(),
                ))

    mean_dice = float(np.nanmean(dices)) if len(dices) else 0.0
    cls_acc = cls_correct / max(cls_total, 1)
    native_metrics = None
    if native_combo_expected_ids is not None:
        from reproduction import aggregate_native_cases

        native_metrics = aggregate_native_cases(
            native_cases, native_combo_expected_ids
        )
    return float(np.mean(val_losses)), mean_dice, cls_acc, native_metrics


#def fit(model, train_loader, val_loader, device,
#        max_epochs=1000, lambda_cls=0.1,
#        initial_lr=1e-2, ckpt_path="best.pth",
#        patience=None, batch_dice=True, optimizer_name='sgd'):

def fit(model, train_loader, val_loader, device,
        max_epochs=1000, lambda_cls=0.1,
        initial_lr=1e-2, ckpt_path="best.pth",
        patience=None, batch_dice=True, optimizer_name='sgd',
        scheduler='poly', warmup_epochs=0, loss_name='dicece',
        wandb_run=None, reproduction_expected_ids=None,
        return_details=False, gradient_accumulation_steps=1):

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
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_cls_acc': []}
    plot_path = os.path.splitext(ckpt_path)[0] + "_curves.png"

    best_score, best_epoch, since_improve = -1.0, -1, 0
    best_native_metrics = None
    epoch_records = []
    global_step = 0
    total_started = time.perf_counter()
    for epoch in range(max_epochs):
        epoch_started = time.perf_counter()
        epoch_start_step = global_step
        #for g in optimizer.param_groups:
        #    g['lr'] = poly_lr(epoch, max_epochs, initial_lr)
        for g in optimizer.param_groups:
            g['lr'] = compute_lr(epoch, max_epochs, initial_lr,
                                 scheduler, warmup_epochs)        

        tr_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, seg_loss_fn,
            lambda_cls, device, scaler, wandb_run=wandb_run,
            epoch=epoch, global_step=global_step,
            gradient_accumulation_steps=gradient_accumulation_steps)
        validation_started = time.perf_counter()
        val_loss, val_dice, val_acc, native_metrics = validate(
            model, val_loader, seg_loss_fn, lambda_cls, device,
            wandb_run=wandb_run, epoch=epoch,
            native_combo_expected_ids=reproduction_expected_ids)
        if native_metrics is not None:
            val_dice = native_metrics['dice']
            val_acc = native_metrics['classification_accuracy']
        validation_seconds = time.perf_counter() - validation_started

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
        if improved:
            best_score, best_epoch, since_improve = final_score, epoch, 0
            best_native_metrics = native_metrics
            save_ckpt(ckpt_path, model, optimizer, epoch, final_score)
        
        else:
            since_improve += 1

        epoch_seconds = time.perf_counter() - epoch_started
        optimizer_steps = global_step - epoch_start_step
        micro_steps = (
            len(train_loader) // gradient_accumulation_steps
        ) * gradient_accumulation_steps
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': tr_loss,
            'validation_loss': val_loss,
            'classification_accuracy': val_acc,
            'dice': val_dice,
            'weighted_composite': final_score,
            'runtime_seconds': epoch_seconds,
            'validation_seconds': validation_seconds,
            'optimizer_steps': optimizer_steps,
            'micro_steps': micro_steps,
            'completed_train_steps': global_step,
        }
        if native_metrics is not None:
            epoch_record['coverage'] = native_metrics['coverage']
        epoch_records.append(epoch_record)
        if wandb_run is not None:
            epoch_log = {
                'epoch': epoch + 1,
                'epoch/train_loss': tr_loss,
                'epoch/validation_loss': val_loss,
                'epoch/classification_accuracy': val_acc,
                'epoch/dice': val_dice,
                'epoch/weighted_composite': final_score,
                'epoch/runtime_seconds': epoch_seconds,
                'epoch/validation_seconds': validation_seconds,
                'epoch/validation_case_count': len(val_loader.dataset),
                'epoch/optimizer_steps': optimizer_steps,
                'epoch/micro_steps': micro_steps,
                'epoch/gradient_accumulation_steps': gradient_accumulation_steps,
                'epoch/completed_train_steps': global_step,
            }
            if native_metrics is not None:
                coverage = native_metrics['coverage']
                epoch_log.update({
                    'epoch/validation_expected': coverage['expected'],
                    'epoch/validation_observed': coverage['observed'],
                    'epoch/validation_unique': coverage['unique'],
                    'epoch/validation_missing': len(coverage['missing']),
                    'epoch/validation_unexpected': len(coverage['unexpected']),
                    'epoch/validation_duplicates': coverage['duplicates'],
                })
            wandb_run.log(epoch_log)
            wandb_run.summary['best/epoch'] = best_epoch + 1
            wandb_run.summary['best/weighted_composite'] = best_score
            wandb_run.summary['runtime/total_seconds'] = (
                time.perf_counter() - total_started
            )

        if patience is not None and since_improve >= patience:
            print(f"Early stop at epoch {epoch} "
                  f"(best final {best_score:.4f} @ {best_epoch})")
            break

    print(f"Done. best final {best_score:.4f} @ epoch {best_epoch}")
    print(f"curves saved to {plot_path}")
    if return_details:
        mps_rng = None
        get_mps_rng_state = getattr(torch.mps, "get_rng_state", None)
        if device.type == "mps" and not callable(get_mps_rng_state):
            raise RuntimeError("MPS continuation requires RNG-state support")
        if device.type == "mps" and callable(get_mps_rng_state):
            mps_rng = get_mps_rng_state()
        return {
            'best_score': best_score,
            'best_epoch': best_epoch + 1,
            'best_native_metrics': best_native_metrics,
            'epochs': epoch_records,
            'completed_epochs': len(epoch_records),
            'completed_train_steps': global_step,
            'completed_micro_steps': sum(
                record['micro_steps'] for record in epoch_records
            ),
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'runtime_seconds': time.perf_counter() - total_started,
            'continuation_state': {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': {
                    'kind': scheduler,
                    'warmup_epochs': warmup_epochs,
                    'initial_lr': initial_lr,
                    'next_epoch': len(epoch_records),
                    'last_learning_rates': [
                        group['lr'] for group in optimizer.param_groups
                    ],
                },
                'completed_epochs': len(epoch_records),
                'global_optimizer_updates': global_step,
                'python_rng_state': random.getstate(),
                'numpy_rng_state': np.random.get_state(),
                'torch_rng_state': torch.get_rng_state(),
                'mps_rng_state': mps_rng,
            },
        }
    return best_score, best_epoch


# small null-context for non-cuda
import contextlib


@contextlib.contextmanager
def _nullctx():
    yield
