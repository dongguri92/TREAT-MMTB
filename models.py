"""
models.py
=========
Multi-task model: PlainConvUNet (segmentation) + classification head
on the encoder bottleneck.

Architecture is taken from nnU-Net's plan (the 1024 / Dataset513 plan):
    n_stages=9, features=[32,64,128,256,512,512,512,512,512],
    InstanceNorm2d, LeakyReLU, stride-2 downsampling (first stage stride 1).

We try to import the exact PlainConvUNet from dynamic_network_architectures
(installed alongside nnunetv2). If unavailable (e.g. a fresh local Windows
env), we fall back to an equivalent hand-written U-Net with the same config.
"""

from typing import List
import torch
import torch.nn as nn

# Forward returns (seg, cls_logit) when return_cls=True, else seg only,
# so that an inference loop calling model(x) for segmentation still works.


# ===========================================================================
#  Try the exact nnU-Net architecture first
# ===========================================================================
def _build_plainconv_unet(in_channels: int, num_classes: int,
                          deep_supervision: bool = False):
    """Build PlainConvUNet from dynamic_network_architectures using the
    Dataset513 (1024) plan. Returns the network or raises ImportError."""
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    net = PlainConvUNet(
        input_channels=in_channels,
        n_stages=9,
        features_per_stage=[32, 64, 128, 256, 512, 512, 512, 512, 512],
        conv_op=nn.Conv2d,
        kernel_sizes=[[3, 3]] * 9,
        strides=[[1, 1]] + [[2, 2]] * 8,
        n_conv_per_stage=[2] * 9,
        num_classes=num_classes,
        n_conv_per_stage_decoder=[2] * 8,
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
    )
    return net


# ===========================================================================
#  Fallback: equivalent hand-written U-Net (same config as the plan)
# ===========================================================================
class _ConvBlock(nn.Module):
    """conv -> InstanceNorm -> LeakyReLU, repeated n_conv times."""
    def __init__(self, in_ch, out_ch, n_conv=2, stride=1):
        super().__init__()
        layers = []
        for i in range(n_conv):
            s = stride if i == 0 else 1
            layers += [
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, stride=s,
                          padding=1, bias=True),
                nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True),
                nn.LeakyReLU(inplace=True),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _FallbackUNet(nn.Module):
    """U-Net matching the nnU-Net plan. forward(x) returns a list of skips via
    .encoder and a seg map via .decoder, mirroring PlainConvUNet's interface
    (so the multi-task wrapper below works for both)."""
    def __init__(self, in_channels, num_classes,
                 features=(32, 64, 128, 256, 512, 512, 512, 512, 512)):
        super().__init__()
        self.features = features
        # ---- encoder ----
        self.enc_blocks = nn.ModuleList()
        prev = in_channels
        for i, f in enumerate(features):
            stride = 1 if i == 0 else 2
            self.enc_blocks.append(_ConvBlock(prev, f, n_conv=2, stride=stride))
            prev = f
        # ---- decoder ----
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.up_convs.append(
                nn.ConvTranspose2d(features[i], features[i - 1], 2, stride=2))
            self.dec_blocks.append(
                _ConvBlock(features[i - 1] * 2, features[i - 1], n_conv=2))
        self.seg_head = nn.Conv2d(features[0], num_classes, 1)

        # expose .encoder/.decoder-like accessors used by the wrapper
        self.out_channels_bottleneck = features[-1]

    def encode(self, x) -> List[torch.Tensor]:
        skips = []
        for blk in self.enc_blocks:
            x = blk(x)
            skips.append(x)
        return skips

    def decode(self, skips) -> torch.Tensor:
        x = skips[-1]
        for j, (up, dec) in enumerate(zip(self.up_convs, self.dec_blocks)):
            x = up(x)
            skip = skips[-(j + 2)]
            # pad if odd sizes mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:],
                                              mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.seg_head(x)

    def forward(self, x):
        skips = self.encode(x)
        return self.decode(skips)


# ===========================================================================
#  Multi-task wrapper
# ===========================================================================
class MultiTaskUNet(nn.Module):
    """
    Wraps a segmentation U-Net (PlainConvUNet or fallback) and adds a
    classification head on the bottleneck feature.

    forward(x):
        return_cls=True  -> (seg, cls_logit)
        return_cls=False -> seg            (for plain inference)
    """
    def __init__(self, in_channels=1, num_classes=2,
                 deep_supervision=False, dropout_cls=0.5):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.return_cls = True
        self._is_plainconv = False

        try:
            self.base = _build_plainconv_unet(in_channels, num_classes,
                                              deep_supervision)
            self._is_plainconv = True
            bottleneck_ch = 512  # last of features_per_stage
        except Exception as e:  # ImportError or any build error -> fallback
            print(f"[models] PlainConvUNet unavailable ({e}); "
                  f"using fallback U-Net.")
            self.base = _FallbackUNet(in_channels, num_classes)
            bottleneck_ch = self.base.out_channels_bottleneck
        
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_cls),
            nn.Linear(bottleneck_ch, 1),
        )

    # -- get encoder skips + seg from whichever base we have ----------------
    def _forward_base(self, x):
        if self._is_plainconv:
            skips = self.base.encoder(x)
            seg = self.base.decoder(skips)
            bottleneck = skips[-1]
        else:
            skips = self.base.encode(x)
            seg = self.base.decode(skips)
            bottleneck = skips[-1]
        return seg, bottleneck

    def forward(self, x):
        seg, bottleneck = self._forward_base(x)
        if not self.return_cls:
            return seg
        cls_logit = self.cls_head(bottleneck)
        return seg, cls_logit


# ===========================================================================
#  Model selector (kept in your style)
# ===========================================================================
def modeltype(model: str, in_channels: int = 1, deep_supervision: bool = False,
              img_size: int = 1024, pretrained_path: str = None,
              variant: str = "small"):
    if model == 'multitask_unet':
        return MultiTaskUNet(in_channels=in_channels, num_classes=2,
                             deep_supervision=deep_supervision)
    elif model == 'evax_seg':
        from models_evax import EVAXSegNet
#        return EVAXSegNet(in_channels=in_channels, num_classes=2,
#                          img_size=img_size, pretrained_path=pretrained_path)
        return EVAXSegNet(in_channels=in_channels, num_classes=2,
                          img_size=img_size, pretrained_path=pretrained_path,
                          variant=variant)
    else:
        raise ValueError(f"Unknown model: {model}")