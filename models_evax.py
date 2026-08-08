"""
models_evax.py — EVA-X (CXR foundation model) backbone + segmentation decoder
=============================================================================
MultiTaskUNet과 동일한 인터페이스를 제공하므로 datasets.py / training.py /
inference.py / Docker를 그대로 재사용할 수 있다.

    model = EVAXSegNet(in_channels=1, num_classes=2, img_size=1024,
                       pretrained_path="~/eva_x_backup/eva_x_small_patch16_merged520k_mim.pt")
    model.return_cls = True   # (seg, cls_logit) 반환  -- MultiTaskUNet과 동일
    model.return_cls = False  # seg만 반환

구조:
    EVA-X small ViT (patch16, embed 384, depth 12, heads 6, SwiGLU, RoPE)
      -> forward_intermediates(indices=[2,5,8,11], output_fmt='NCHW')
         : 4개 모두 H/16 해상도 (ViT는 downsampling이 없음)
      -> Simple Feature Pyramid (ViTDet 방식): H/4, H/8, H/16, H/32로 재조정
      -> FPN top-down fusion -> H/4에서 병합 -> seg head -> 입력 해상도로 upsample

mmcv / mmsegmentation 불필요. timm >= 0.9 (검증: 1.0.22) 만 있으면 된다.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy._core.multiarray import scalar as numpy_scalar

from timm.models.eva import Eva


# =============================================================================
#  EVA-X backbone
# =============================================================================
def _adapt_patch_embed_in_chans(state_dict, in_chans):
    """사전학습 patch_embed의 입력 채널수를 우리 입력(1채널)에 맞춘다.
    3->1 은 채널 합산 (grayscale을 3채널로 복제했을 때와 동일한 응답)."""
    k = "patch_embed.proj.weight"
    if k not in state_dict:
        return state_dict
    w = state_dict[k]
    c = w.shape[1]
    if c == in_chans:
        return state_dict
    if in_chans == 1:
        state_dict[k] = w.sum(dim=1, keepdim=True)
    else:
        rep = int(math.ceil(in_chans / c))
        state_dict[k] = w.repeat(1, rep, 1, 1)[:, :in_chans]
    print(f"  patch_embed in_chans {c} -> {in_chans}")
    return state_dict

#def build_eva_x_small(img_size=1024, in_chans=1, pretrained_path=None):
#    model = Eva(
#        img_size=img_size,
#        patch_size=16,
#        in_chans=in_chans,
#        num_classes=0,
#        embed_dim=384,
#        depth=12,
#        num_heads=6,
#        mlp_ratio=4 * 2 / 3,
#        swiglu_mlp=True,
#        use_rot_pos_emb=True,
#        ref_feat_shape=(14, 14),      # 224/16 — 사전학습 grid
#        dynamic_img_size=True,
#    )

def build_eva_x_small(img_size=1024, in_chans=1, pretrained_path=None,
                      variant="small"):
    """EVA-X backbone. variant: 'small'(embed384) | 'base'(embed768).
    base는 eva_x.py의 eva_x_base_patch16과 동일하게 qkv 분리 + scale_mlp 사용."""
    if variant == "base":
        arch = dict(embed_dim=768, depth=12, num_heads=12,
                    qkv_fused=False, scale_mlp=True)
    else:
        arch = dict(embed_dim=384, depth=12, num_heads=6)

    model = Eva(
        img_size=img_size,
        patch_size=16,
        in_chans=in_chans,
        num_classes=0,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),
        dynamic_img_size=True,
        drop_path_rate=0.1,
        **arch,
    )

    if pretrained_path:
        import os
        from eva_x import checkpoint_filter_fn      # 저장소에서 가져온 파일

        path = os.path.expanduser(pretrained_path)
        numpy_safe_globals = [
            (numpy_scalar, "numpy.core.multiarray.scalar"),
            np.dtype,
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(numpy_safe_globals):
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint_filter_fn(ckpt, model)   # pos_embed / patch_embed resample
        state = _adapt_patch_embed_in_chans(state, in_chans)
        msg = model.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(msg.unexpected_keys)
        print(f"  EVA-X pretrained: {path}")
        print(f"    loaded {n_loaded} tensors | "
              f"missing {len(msg.missing_keys)} | unexpected {len(msg.unexpected_keys)}")
        if len(msg.missing_keys) > 20:
            print(f"    [WARN] missing이 많음 — 로드 실패 가능: {msg.missing_keys[:5]} ...")
    else:
        print("  [WARN] pretrained_path 없음 — random init (foundation model 이점 없음)")

    return model


# =============================================================================
#  Simple Feature Pyramid (ViTDet) + FPN decoder
# =============================================================================
class _ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
        )


class SimpleFeaturePyramid(nn.Module):
    """단일 해상도(H/16) ViT feature 4개를 H/4, H/8, H/16, H/32로 재조정.
    ViTDet / UperNet-on-ViT에서 쓰는 표준 방식."""

    def __init__(self, embed_dim=384, out_ch=256):
        super().__init__()
        # H/16 -> H/4   (x4 up)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, out_ch, 2, 2),
        )
        # H/16 -> H/8   (x2 up)
        self.up2 = nn.ConvTranspose2d(embed_dim, out_ch, 2, 2)
        # H/16 -> H/16
        self.id1 = nn.Conv2d(embed_dim, out_ch, 1)
        # H/16 -> H/32  (x2 down)
        self.down2 = nn.Conv2d(embed_dim, out_ch, 3, 2, 1)

    def forward(self, feats):
        f0, f1, f2, f3 = feats
        return [self.up4(f0), self.up2(f1), self.id1(f2), self.down2(f3)]


class FPNDecoder(nn.Module):
    """top-down fusion 후 모든 레벨을 H/4로 모아 병합."""

    def __init__(self, ch=256, num_classes=2):
        super().__init__()
        self.smooth = nn.ModuleList([_ConvBNAct(ch, ch) for _ in range(4)])
        self.fuse = _ConvBNAct(ch * 4, ch)
        self.head = nn.Sequential(
            _ConvBNAct(ch, ch // 2),
            nn.Conv2d(ch // 2, num_classes, 1),
        )

    def forward(self, pyr, out_size):
        # top-down: 낮은 해상도부터 위로 더해 올림
        x = pyr[3]
        outs = [x]
        for i in (2, 1, 0):
            x = pyr[i] + F.interpolate(x, size=pyr[i].shape[-2:],
                                       mode="bilinear", align_corners=False)
            outs.append(x)
        outs = outs[::-1]                       # [H/4, H/8, H/16, H/32]
        outs = [s(o) for s, o in zip(self.smooth, outs)]

        target = outs[0].shape[-2:]             # H/4
        outs = [outs[0]] + [F.interpolate(o, size=target, mode="bilinear",
                                          align_corners=False)
                            for o in outs[1:]]
        x = self.fuse(torch.cat(outs, dim=1))
        x = self.head(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


# =============================================================================
#  EVAXSegNet — MultiTaskUNet과 동일 인터페이스
# =============================================================================
class EVAXSegNet(nn.Module):
#    def __init__(self, in_channels=1, num_classes=2, img_size=1024,
#                 pretrained_path=None, decoder_ch=256,
#                 indices=(2, 5, 8, 11), dropout_cls=0.5):
    def __init__(self, in_channels=1, num_classes=2, img_size=1024,
                 pretrained_path=None, decoder_ch=256,
                 indices=(2, 5, 8, 11), dropout_cls=0.5, variant="small"):
        super().__init__()
        self.return_cls = True
        self.indices = list(indices)

        self.backbone = build_eva_x_small(img_size=img_size,
                                          in_chans=in_channels,
                                          pretrained_path=pretrained_path,
                                          variant=variant)
        embed_dim = self.backbone.embed_dim

        self.pyramid = SimpleFeaturePyramid(embed_dim, decoder_ch)
        self.decoder = FPNDecoder(decoder_ch, num_classes)

        # MultiTaskUNet 인터페이스 호환용. --lambda_cls 0 이면 학습되지 않고
        # inference.py --detection seg 에서도 쓰이지 않음.
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_cls),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        out_size = x.shape[-2:]
        feats = self.backbone.forward_intermediates(
            x, indices=self.indices, output_fmt="NCHW", intermediates_only=True)
        seg = self.decoder(self.pyramid(feats), out_size)

        if not self.return_cls:
            return seg
        cls_logit = self.cls_head(feats[-1])
        return seg, cls_logit

    # ViT fine-tuning용 param group: backbone은 낮은 lr, decoder는 높은 lr
    def param_groups(self, lr, backbone_lr_mult=0.1, weight_decay=0.05):
        decay, no_decay = [], []
        for n, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
        head_params = list(self.pyramid.parameters()) + \
            list(self.decoder.parameters()) + list(self.cls_head.parameters())
        return [
            {"params": decay, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr * backbone_lr_mult, "weight_decay": 0.0},
            {"params": head_params, "lr": lr, "weight_decay": weight_decay},
        ]
