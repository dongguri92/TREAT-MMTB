import os
import argparse
import torch

from models import modeltype
from training import fit


CONFIG = dict(
    target_size=1024,
    clahe_clip=2.0,

    seed=42,

    max_epochs=1000,
    batch_size=3,
    lambda_cls=0.3,
    initial_lr=1e-2,
    num_workers=4,
    deep_supervision=False,
    batch_dice=True,
    patience=None,

    ckpt_path="best_mtl_fold0.pth",
)

EVAX_PRETRAINED = "~/eva_x_backup/eva_x_small_patch16_merged520k_mim.pt"
EVAX_PRETRAINED_BASE = "~/eva_x_backup/eva_x_base_patch16_merged520k_mim.pt"


def main():
    cfg = dict(CONFIG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='multitask_unet',
                        choices=['multitask_unet', 'evax_seg'])
    parser.add_argument('--pretrained', type=str, default=None,
                        help="EVA-X 사전학습 가중치 경로 (evax_seg 전용). "
                             "미지정 시 기본 경로 사용")
    parser.add_argument('--optimizer', type=str, default=None,
                        choices=['sgd', 'adamw'],
                        help="미지정 시 multitask_unet->sgd, evax_seg->adamw")
    parser.add_argument('--channels', type=int, default=1, choices=[1, 3],
                        help="1: datasets.py (원본1채널) / 3: datasets_3ch.py "
                             "(원본/CLAHE2.0/CLAHE1.0)")
    parser.add_argument('--lambda_cls', type=float, default=cfg['lambda_cls'])
    parser.add_argument('--target_size', type=int, default=cfg['target_size'],
                        help="resolution, e.g. 512 / 768 / 1024")
    parser.add_argument('--batch_size', type=int, default=cfg['batch_size'])
    parser.add_argument('--initial_lr', type=float, default=None,
                        help="미지정 시 sgd->1e-2, adamw->1e-4")
    parser.add_argument('--max_epochs', type=int, default=cfg['max_epochs'])
    parser.add_argument('--num_workers', type=int, default=cfg['num_workers'])
    parser.add_argument('--tag', type=str, default=None,
                        help="save tag, e.g. evax -> best_evax.pth")
    parser.add_argument('--scheduler', type=str, default='poly',
                        choices=['poly', 'cosine'])
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--loss', type=str, default='dicece',
                        choices=['dicece', 'tversky', 'boundary'])
    parser.add_argument('--variant', type=str, default='small',
                        choices=['small', 'base'])
    parser.add_argument('--crop_frac', type=float, default=0.15,
                        help="하부 crop 비율 (0.15=기존, 0.20=더 자름, 0=crop 없음)")
    parser.add_argument('--roi', action='store_true',
                        help="ROI 2단계 학습 (datasets_roi 사용, 양성 성분별 crop)")
    parser.add_argument('--roi_context', type=float, default=1.5)
    parser.add_argument('--roi_jitter', type=float, default=0.15,
                        help="ROI 중심 이동 최대 비율")
    parser.add_argument('--size_weighted', action='store_true',
                        help="cavity 크기별 가중 샘플링 (small 3x, medium 2x)")
    args = parser.parse_args()

    cfg['lambda_cls'] = args.lambda_cls
    cfg['target_size'] = args.target_size
    cfg['batch_size'] = args.batch_size
    cfg['max_epochs'] = args.max_epochs
    cfg['num_workers'] = args.num_workers
    if args.tag is not None:
        cfg['ckpt_path'] = f"best_{args.tag}.pth"

    # optimizer / lr 기본값을 모델에 맞춰 결정
    is_vit = (args.model == 'evax_seg')
    opt = args.optimizer or ('adamw' if is_vit else 'sgd')
    if args.initial_lr is not None:
        cfg['initial_lr'] = args.initial_lr
    else:
        cfg['initial_lr'] = 1e-4 if opt == 'adamw' else 1e-2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model={args.model} | channels={args.channels} | "
          f"target_size={cfg['target_size']} | "
          f"batch_size={cfg['batch_size']} | lambda_cls={cfg['lambda_cls']} | "
          f"opt={opt} lr={cfg['initial_lr']} | ckpt={cfg['ckpt_path']}"
          f"crop={args.crop_frac} | ")

    # 채널 수에 따라 dataloader 모듈 선택
    if args.roi:
        from datasets_roi import dataloader
        train_loader, val_loader = dataloader(
            batch_size=cfg['batch_size'], roi_size=cfg['target_size'],
            context=args.roi_context, clahe_clip=cfg['clahe_clip'],
            num_workers=cfg['num_workers'], seed=cfg['seed'],
            jitter=(args.roi_jitter, 0.8, 1.4), per_component=True)
    elif args.channels == 3:
        from datasets_3ch import dataloader
        train_loader, val_loader = dataloader(
            batch_size=cfg['batch_size'],
            target_size=cfg['target_size'], clahe_clip=cfg['clahe_clip'],
            num_workers=cfg['num_workers'], seed=cfg['seed'],
            crop_frac=args.crop_frac)
    else:
        from datasets import dataloader
        train_loader, val_loader = dataloader(
            batch_size=cfg['batch_size'],
            target_size=cfg['target_size'], clahe_clip=cfg['clahe_clip'],
            num_workers=cfg['num_workers'], seed=cfg['seed'],
            crop_frac=args.crop_frac,
            size_weighted=args.size_weighted)

    if is_vit:
        default_pt = (EVAX_PRETRAINED_BASE if args.variant == 'base'
                      else EVAX_PRETRAINED)
        pretrained = args.pretrained or default_pt
        model = modeltype('evax_seg', in_channels=args.channels,
                          img_size=cfg['target_size'],
                          pretrained_path=pretrained,
                          variant=args.variant).to(device)
    else:
        model = modeltype('multitask_unet', in_channels=args.channels,
                          deep_supervision=cfg['deep_supervision']).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.1f}M")

    fit(model, train_loader, val_loader, device,
        max_epochs=cfg['max_epochs'],
        lambda_cls=cfg['lambda_cls'],
        initial_lr=cfg['initial_lr'],
        ckpt_path=cfg['ckpt_path'],
        patience=cfg['patience'],
        batch_dice=cfg['batch_dice'],
        optimizer_name=opt,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup,
        loss_name=args.loss)


if __name__ == "__main__":
    main()