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


def _fp32_for_loss(value):
    """Promote real tensor outputs while keeping lightweight test doubles valid."""
    return value.float() if isinstance(value, torch.Tensor) else value


def accumulated_train_step(
    model,
    batches,
    optimizer,
    seg_loss_fn,
    lambda_cls,
    device,
    scaler=None,
    use_amp=True,
    critical_memory_observer=None,
    critical_memory_context=None,
    stage_callback=None,
    accumulation=None,
    batch_observer=None,
    group_validator=None,
    amp_dtype=None,
):
    """Run one historical attempt-4 accumulated optimizer sequence."""
    if accumulation is None:
        accumulation = len(batches)
    if accumulation < 1:
        raise ValueError("accumulated step requires at least one microbatch")
    bce = torch.nn.BCEWithLogitsLoss()
    rows = []
    deferred_last = None

    def materialize_row(values):
        total, segmentation, classification = values
        total_value = float(total.item())
        # Preserve attempt-4's W&B-backed scalar schedule exactly: the total
        # loss was materialized once for history and twice for batch/total logs.
        float(total.item())
        float(total.item())
        return {
            'loss': total_value,
            'segmentation_loss': float(segmentation.item()),
            'classification_loss': float(classification.item()),
        }
    iterator = iter(batches)
    for microbatch_index in range(accumulation):
        batch = next(iterator)
        if batch_observer is not None:
            batch_observer(batch, microbatch_index)
        img = batch['image'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        cls = batch['cls'].to(device, non_blocking=True)
        with _autocast_context(device, use_amp, amp_dtype):
            model.return_cls = True
            if stage_callback is not None:
                stage_callback("forward")
            seg_out, cls_logit = model(img)
        if stage_callback is not None:
            stage_callback("loss")
        # Loss math is deliberately outside autocast and uses FP32 outputs.
        loss_seg_out = (
            [_fp32_for_loss(value) for value in seg_out]
            if isinstance(seg_out, (list, tuple))
            else _fp32_for_loss(seg_out)
        )
        l_seg = _seg_loss_on_output(loss_seg_out, mask, seg_loss_fn)
        l_cls = bce(_fp32_for_loss(cls_logit), _fp32_for_loss(cls))
        loss = l_seg + lambda_cls * l_cls
        backward_loss = loss / accumulation
        if scaler is not None:
            if stage_callback is not None:
                stage_callback("backward")
            scaler.scale(backward_loss).backward()
        else:
            if stage_callback is not None:
                stage_callback("backward")
            backward_loss.backward()
        detached = (loss.detach(), l_seg.detach(), l_cls.detach())
        if microbatch_index < accumulation - 1:
            if stage_callback is not None:
                stage_callback("scalar_materialization_between_microbatches")
            rows.append(materialize_row(detached))
        else:
            deferred_last = detached

    if group_validator is not None:
        group_validator()

    if stage_callback is not None:
        stage_callback("critical_memory")
    critical_memory = None
    if critical_memory_observer is not None:
        critical_memory = (
            critical_memory_observer(critical_memory_context)
            if critical_memory_context is not None
            else critical_memory_observer()
        )
    if stage_callback is not None:
        stage_callback("gradient_clip")
    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
    if scaler is not None:
        if stage_callback is not None:
            stage_callback("optimizer_step")
        scaler.step(optimizer)
        scaler.update()
    else:
        if stage_callback is not None:
            stage_callback("optimizer_step")
        optimizer.step()
    if stage_callback is not None:
        stage_callback("optimizer_zero_grad_after")
    optimizer.zero_grad(set_to_none=True)

    if stage_callback is not None:
        stage_callback("scalar_materialization")
    if deferred_last is None:
        raise RuntimeError("accumulated step did not produce a final microbatch")
    rows.append(materialize_row(deferred_last))
    if not all(
        math.isfinite(value)
        for row in rows
        for value in row.values()
    ):
        raise FloatingPointError("accumulated optimizer step produced non-finite values")
    return {
        'microbatches': rows,
        'critical_memory': critical_memory,
    }


def train_one_epoch(model, loader, optimizer, seg_loss_fn, lambda_cls,
                    device, scaler, use_amp=True, wandb_run=None,
                    epoch=0, global_step=0, gradient_accumulation_steps=1,
                    progress_callback=None, critical_memory_observer=None,
                    first_group_batch_observer=None,
                    first_group_validator=None, amp_dtype=None):
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    usable_micro_steps = (
        len(loader) // gradient_accumulation_steps
    ) * gradient_accumulation_steps
    if usable_micro_steps == 0:
        raise ValueError("loader cannot provide one complete effective batch")
    model.train()
    losses = []
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    for group_start in range(0, usable_micro_steps, gradient_accumulation_steps):
        batches = (
            next(iterator) for _ in range(gradient_accumulation_steps)
        )
        step = accumulated_train_step(
            model,
            batches,
            optimizer,
            seg_loss_fn,
            lambda_cls,
            device,
            scaler=scaler,
            use_amp=use_amp,
            critical_memory_observer=critical_memory_observer,
            critical_memory_context={
                'epoch': epoch + 1,
                'optimizer_step_in_epoch': (
                    group_start // gradient_accumulation_steps + 1
                ),
                'global_optimizer_step': global_step + 1,
            },
            accumulation=gradient_accumulation_steps,
            batch_observer=(
                first_group_batch_observer if group_start == 0 else None
            ),
            group_validator=(
                first_group_validator if group_start == 0 else None
            ),
            amp_dtype=amp_dtype,
        )
        global_step += 1
        for offset, row in enumerate(step['microbatches']):
            batch_index = group_start + offset
            losses.append(row['loss'])
            optimizer_step_completed = offset == gradient_accumulation_steps - 1
            if wandb_run is not None:
                step_log = {
                    'train/global_step': global_step,
                    'train/micro_step': epoch * usable_micro_steps + batch_index + 1,
                    'train/epoch': epoch + 1,
                    'train/batch_loss': row['loss'],
                    'train/total_loss': row['loss'],
                    'train/segmentation_loss': row['segmentation_loss'],
                    'train/classification_loss': row['classification_loss'],
                    'train/optimizer_step_completed': optimizer_step_completed,
                    'train/gradient_accumulation_steps': gradient_accumulation_steps,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                }
                step_log.update({
                    f'train/learning_rate_group_{index}': group['lr']
                    for index, group in enumerate(optimizer.param_groups)
                })
                wandb_run.log(step_log)
            if progress_callback is not None:
                progress_callback({
                    'phase': 'scientific_train',
                    'epoch': epoch + 1,
                    'micro_step': batch_index + 1,
                    'optimizer_step': global_step,
                    'optimizer_step_completed': optimizer_step_completed,
                })
    return float(np.mean(losses)), global_step


def validation_step(model, batch, seg_loss_fn, lambda_cls, device,
                    use_amp=True, materialize_log_loss=False,
                    native_case_materialization=False, stage_callback=None,
                    amp_dtype=None):
    """Execute the scientific validation tensor/host boundary for one batch."""
    bce = torch.nn.BCEWithLogitsLoss()
    if stage_callback is not None:
        stage_callback('device_transfers')
    img = batch['image'].to(device, non_blocking=True)
    mask = batch['mask'].to(device, non_blocking=True)
    cls = batch['cls'].to(device, non_blocking=True)
    with _autocast_context(device, use_amp, amp_dtype):
        model.return_cls = True
        if stage_callback is not None:
            stage_callback('forward')
        seg_out, cls_logit = model(img)
        seg_main = seg_out[0] if isinstance(seg_out, (list, tuple)) else seg_out
    l_seg = seg_loss_fn(_fp32_for_loss(seg_main), mask)
    l_cls = bce(_fp32_for_loss(cls_logit), _fp32_for_loss(cls))
    loss = l_seg + lambda_cls * l_cls
    if stage_callback is not None:
        stage_callback('loss_item')
    loss_value = loss.item()
    log_loss_value = None
    if materialize_log_loss:
        if stage_callback is not None:
            stage_callback('log_loss_item')
        log_loss_value = loss.item()
    if stage_callback is not None:
        stage_callback('prepared_masks_cpu')
    pred = seg_main.float().argmax(1).cpu().numpy()
    gt = mask.squeeze(1).cpu().numpy()
    dices = [dice_metric(pred[index], gt[index]) for index in range(pred.shape[0])]
    cls_pred = (torch.sigmoid(cls_logit.float()) > 0.5).float()
    if stage_callback is not None:
        stage_callback('classification_sum_item')
    cls_correct = (cls_pred == cls).sum().item()
    cls_total = cls.numel()
    native_cases = []
    if native_case_materialization:
        from reproduction import native_case_record

        required = {'id', 'native_mask', 'native_shape', 'crop_shape', 'pad_info'}
        missing_fields = sorted(required - set(batch))
        if missing_fields:
            raise ValueError(f"native validation metadata missing: {missing_fields}")
        if stage_callback is not None:
            stage_callback('probabilities_cpu')
        fg_prob = torch.softmax(seg_main.float(), dim=1)[:, 1].cpu().numpy()
        cls_prob = torch.sigmoid(cls_logit.float()).flatten().cpu().numpy()
        for index, case_id in enumerate(batch['id']):
            if stage_callback is not None:
                stage_callback('native_metadata_cpu')
            native_cases.append(native_case_record(
                case_id=case_id,
                foreground_probability=fg_prob[index],
                cls_probability=cls_prob[index],
                native_mask=batch['native_mask'][index].cpu().numpy(),
                pad_info=batch['pad_info'][index].cpu().numpy(),
                crop_shape=batch['crop_shape'][index].cpu().numpy(),
                native_shape=batch['native_shape'][index].cpu().numpy(),
            ))
    return {
        'loss': loss_value, 'log_loss': log_loss_value, 'dices': dices,
        'cls_correct': cls_correct, 'cls_total': cls_total,
        'native_cases': native_cases,
    }


@torch.no_grad()
def validate(model, loader, seg_loss_fn, lambda_cls, device, use_amp=True,
             wandb_run=None, epoch=0, native_combo_expected_ids=None,
             progress_callback=None, amp_dtype=None):
    model.eval()
    dices, cls_correct, cls_total = [], 0, 0
    native_cases = []
    val_losses = []
    for batch_index, batch in enumerate(loader):
        row = validation_step(
            model, batch, seg_loss_fn, lambda_cls, device, use_amp=use_amp,
            materialize_log_loss=wandb_run is not None,
            native_case_materialization=native_combo_expected_ids is not None,
            amp_dtype=amp_dtype,
        )
        val_losses.append(row['loss'])
        if wandb_run is not None:
            wandb_run.log({
                'validation/global_step': epoch * len(loader) + batch_index + 1,
                'validation/epoch': epoch + 1,
                'validation/batch_loss': row['log_loss'],
            })
        dices.extend(row['dices'])
        cls_correct += row['cls_correct']
        cls_total += row['cls_total']
        native_cases.extend(row['native_cases'])
        if progress_callback is not None:
            progress_callback({
                'phase': 'scientific_validation',
                'epoch': epoch + 1,
                'validation_step': batch_index + 1,
            })

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
        return_details=False, gradient_accumulation_steps=1,
        progress_callback=None, critical_memory_observer=None,
        first_group_batch_observer=None, first_group_validator=None,
        amp_dtype=None):

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
            gradient_accumulation_steps=gradient_accumulation_steps,
            progress_callback=progress_callback,
            critical_memory_observer=critical_memory_observer,
            first_group_batch_observer=(
                first_group_batch_observer if epoch == 0 else None
            ),
            first_group_validator=(
                first_group_validator if epoch == 0 else None
            ), amp_dtype=amp_dtype)
        validation_started = time.perf_counter()
        val_loss, val_dice, val_acc, native_metrics = validate(
            model, val_loader, seg_loss_fn, lambda_cls, device,
            wandb_run=wandb_run, epoch=epoch,
            native_combo_expected_ids=reproduction_expected_ids,
            progress_callback=progress_callback, amp_dtype=amp_dtype)
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
        }
    return best_score, best_epoch


# small null-context for non-cuda
import contextlib


@contextlib.contextmanager
def _nullctx():
    yield


def _autocast_context(device, enabled, amp_dtype=None):
    """Return an explicit autocast context; unsupported requests fail closed."""
    if not enabled:
        return _nullctx()
    # Preserve the pre-existing non-CUDA full-precision behavior unless a
    # precision dtype is explicitly requested by a sealed experiment.
    if amp_dtype is None and device.type != 'cuda':
        return _nullctx()
    if device.type == 'cuda':
        return autocast('cuda', dtype=amp_dtype, enabled=True)
    if device.type == 'mps':
        if amp_dtype is not torch.bfloat16:
            raise RuntimeError(
                'MPS mixed precision requires explicit torch.bfloat16'
            )
        if not torch.backends.mps.is_available():
            raise RuntimeError('requested MPS BF16 but MPS is unavailable')
        return autocast('mps', dtype=torch.bfloat16, enabled=True)
    raise RuntimeError(
        f'mixed precision is unsupported for device type {device.type!r}'
    )
