"""Standalone X-Raydar classifier and mask-ensemble model definitions."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.inception import BasicConv2d, Inception3


class XRaydarInception3(Inception3):
    """Torchvision Inception-v3 with the official one-channel X-Raydar input."""

    def __init__(self, num_classes: int, aux_logits: bool):
        super().__init__(
            num_classes=num_classes,
            aux_logits=aux_logits,
            transform_input=True,
            init_weights=False,
        )
        self.Conv2d_1a_3x3 = BasicConv2d(
            1, 32, kernel_size=3, stride=2
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        # The public X-Raydar implementation expects an already standardized
        # grayscale channel and applies this final 0.5 scale internally.
        return x * 0.5 if self.transform_input else x


class XRaydarCavityBinary(nn.Module):
    """One-logit XNet38 classifier used by the dedicated first stage."""

    def __init__(self):
        super().__init__()
        self.network = XRaydarInception3(num_classes=1, aux_logits=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.network(image)
        if isinstance(output, tuple):
            output = output[0]
        return output.squeeze(1)


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int = 3):
        groups = min(32, cout)
        while cout % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
        )


class PyramidPooling(nn.Module):
    def __init__(self, cin: int = 256, cout: int = 256):
        super().__init__()
        self.scales = (1, 2, 3, 6)
        self.branches = nn.ModuleList(
            [ConvNormAct(cin, cout // 4, 1) for _ in self.scales]
        )
        self.bottleneck = ConvNormAct(cin + cout, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        parts = [x]
        for scale, branch in zip(self.scales, self.branches):
            pooled = F.adaptive_avg_pool2d(x, (scale, scale))
            parts.append(
                F.interpolate(
                    branch(pooled), size, mode="bilinear", align_corners=False
                )
            )
        return self.bottleneck(torch.cat(parts, dim=1))


class UPerDecoder(nn.Module):
    def __init__(self, cin: int = 256, channels: int = 256):
        super().__init__()
        self.laterals = nn.ModuleList(
            [ConvNormAct(cin, channels, 1) for _ in range(3)]
        )
        self.ppm = PyramidPooling(cin, channels)
        self.fpn = nn.ModuleList(
            [ConvNormAct(channels, channels) for _ in range(3)]
        )
        self.bottleneck = ConvNormAct(channels * 4, channels)
        self.output = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(
        self, features: list[torch.Tensor], output_size: tuple[int, int]
    ) -> torch.Tensor:
        laterals = [
            layer(feature)
            for layer, feature in zip(self.laterals, features[:3])
        ]
        laterals.append(self.ppm(features[3]))
        for index in range(3, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index],
                laterals[index - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        pyramid = [
            layer(feature) for layer, feature in zip(self.fpn, laterals[:3])
        ]
        pyramid.append(laterals[3])
        size = pyramid[0].shape[-2:]
        pyramid = [
            feature
            if feature.shape[-2:] == size
            else F.interpolate(
                feature, size, mode="bilinear", align_corners=False
            )
            for feature in pyramid
        ]
        logits = self.output(self.bottleneck(torch.cat(pyramid, dim=1)))
        return F.interpolate(
            logits, output_size, mode="bilinear", align_corners=False
        ).squeeze(1)


class UNetUpBlock(nn.Module):
    """Learned 2x up-convolution, skip fusion, and two convolutions."""

    def __init__(self, cin: int, skip_channels: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)
        self.refine = nn.Sequential(
            ConvNormAct(cout + skip_channels, cout, 3),
            ConvNormAct(cout, cout, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        x = self.up(x)
        target_size = skip.shape[-2:] if skip is not None else output_size
        if target_size is not None and x.shape[-2:] != target_size:
            # Inception valid-stride stages produce odd grids. The learned
            # up-convolution is retained; interpolation only aligns its grid.
            x = F.interpolate(
                x, size=target_size, mode="bilinear", align_corners=False
            )
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        return self.refine(x)


class XRaydarUNetDecoder(nn.Module):
    """Five-level U-Net decoder over the XNet38 feature hierarchy."""

    def __init__(self):
        super().__init__()
        self.bottleneck = ConvNormAct(2048, 512, 3)
        self.blocks = nn.ModuleList(
            [
                UNetUpBlock(512, 768, 256),
                UNetUpBlock(256, 288, 128),
                UNetUpBlock(128, 192, 64),
                UNetUpBlock(64, 64, 32),
                UNetUpBlock(32, 0, 16),
            ]
        )
        self.output = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(
        self, features: list[torch.Tensor], output_size: tuple[int, int]
    ) -> torch.Tensor:
        c0, c1, c2, c3, c4 = features
        x = self.bottleneck(c4)
        for block, skip in zip(self.blocks[:4], (c3, c2, c1, c0)):
            x = block(x, skip=skip)
        x = self.blocks[4](x, output_size=output_size)
        if x.shape[-2:] != output_size:
            x = F.interpolate(
                x, size=output_size, mode="bilinear", align_corners=False
            )
        return self.output(x).squeeze(1)


class XRaydarUNet(nn.Module):
    """XNet38 encoder with the trained positive-only U-Net decoder."""

    def __init__(self):
        super().__init__()
        self.backbone = XRaydarInception3(num_classes=38, aux_logits=True)
        self.classifier = nn.Linear(self.backbone.fc.in_features, 1)
        del self.backbone.fc
        del self.backbone.AuxLogits
        self.backbone.aux_logits = False

        self.decoder = XRaydarUNetDecoder()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.backbone._transform_input(image)
        x = self.backbone.Conv2d_1a_3x3(x)
        x = self.backbone.Conv2d_2a_3x3(x)
        x = self.backbone.Conv2d_2b_3x3(x)
        c0 = x
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.backbone.Conv2d_3b_1x1(x)
        c1 = self.backbone.Conv2d_4a_3x3(x)

        x = F.max_pool2d(c1, kernel_size=3, stride=2)
        x = self.backbone.Mixed_5b(x)
        x = self.backbone.Mixed_5c(x)
        c2 = self.backbone.Mixed_5d(x)

        x = self.backbone.Mixed_6a(c2)
        x = self.backbone.Mixed_6b(x)
        x = self.backbone.Mixed_6c(x)
        x = self.backbone.Mixed_6d(x)
        c3 = self.backbone.Mixed_6e(x)

        x = self.backbone.Mixed_7a(c3)
        x = self.backbone.Mixed_7b(x)
        c4 = self.backbone.Mixed_7c(x)
        return self.decoder([c0, c1, c2, c3, c4], image.shape[-2:])


def _m2f_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class M2FConvNormAct(nn.Sequential):
    """Convolution block used by the trained Mask2Former pixel decoder."""

    def __init__(self, cin: int, cout: int, kernel: int):
        super().__init__(
            nn.Conv2d(
                cin,
                cout,
                kernel,
                padding=kernel // 2,
                bias=False,
            ),
            _m2f_group_norm(cout),
            nn.ReLU(inplace=True),
        )


class XRaydarFPNPixelDecoder(nn.Module):
    """Four-level FPN pixel decoder used by the trained Mask2Former."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        feature_channels = (192, 288, 768, 2048)
        self.top = M2FConvNormAct(feature_channels[-1], hidden_dim, 3)
        self.laterals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden_dim, 1, bias=False),
                    _m2f_group_norm(hidden_dim),
                )
                for channels in feature_channels[:-1]
            ]
        )
        self.outputs = nn.ModuleList(
            [M2FConvNormAct(hidden_dim, hidden_dim, 3) for _ in range(3)]
        )
        self.mask_features = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1
        )

    def forward(self, features: list[torch.Tensor]):
        c2, c3, c4, c5 = features
        y = self.top(c5)
        multi_scale = [y]
        for index, feature in reversed(list(enumerate((c2, c3, c4)))):
            lateral = self.laterals[index](feature)
            y = self.outputs[index](
                lateral
                + F.interpolate(
                    y, size=lateral.shape[-2:], mode="nearest"
                )
            )
            if len(multi_scale) < 3:
                multi_scale.append(y)
        return self.mask_features(y), multi_scale


class PositionEmbeddingSine(nn.Module):
    def __init__(self, features: int = 128, temperature: int = 10000):
        super().__init__()
        self.features = features
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        valid = torch.ones(
            x.shape[0],
            x.shape[-2],
            x.shape[-1],
            dtype=torch.bool,
            device=x.device,
        )
        y_embed = valid.cumsum(1, dtype=torch.float32)
        x_embed = valid.cumsum(2, dtype=torch.float32)
        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale
        dim = torch.arange(
            self.features, dtype=torch.float32, device=x.device
        )
        dim = self.temperature ** (
            2 * torch.div(dim, 2, rounding_mode="floor") / self.features
        )
        pos_x = x_embed[..., None] / dim
        pos_y = y_embed[..., None] / dim
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class M2FSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int = 256, heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        self.dropout = nn.Dropout(0.0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, query_pos: torch.Tensor):
        query = x + query_pos
        update = self.attention(query, query, x, need_weights=False)[0]
        return self.norm(x + self.dropout(update))


class M2FCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int = 256, heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        self.dropout = nn.Dropout(0.0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        memory_pos: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        update = self.attention(
            x + query_pos,
            memory + memory_pos,
            memory,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        return self.norm(x + self.dropout(update))


class M2FFeedForward(nn.Module):
    def __init__(self, hidden_dim: int = 256, feedforward_dim: int = 2048):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.dropout = nn.Dropout(0.0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        update = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.norm(x + self.dropout(update))


class M2FMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        layers: int = 3,
    ):
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [
                nn.Linear(dimensions[index], dimensions[index + 1])
                for index in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index + 1 != len(self.layers):
                x = F.relu(x)
        return x


class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 15,
        heads: int = 8,
        decoder_layers: int = 9,
    ):
        super().__init__()
        self.num_heads = heads
        self.num_feature_levels = 3
        self.position = PositionEmbeddingSine(hidden_dim // 2)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.query_feature = nn.Embedding(num_queries, hidden_dim)
        self.query_position = nn.Embedding(num_queries, hidden_dim)
        self.cross_layers = nn.ModuleList(
            [M2FCrossAttention(hidden_dim, heads) for _ in range(decoder_layers)]
        )
        self.self_layers = nn.ModuleList(
            [M2FSelfAttention(hidden_dim, heads) for _ in range(decoder_layers)]
        )
        self.ffn_layers = nn.ModuleList(
            [M2FFeedForward(hidden_dim) for _ in range(decoder_layers)]
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, 2)
        self.mask_embed = M2FMLP(hidden_dim, hidden_dim, hidden_dim, 3)

    def prediction_heads(
        self,
        output: torch.Tensor,
        mask_features: torch.Tensor,
        attention_size: tuple[int, int],
    ):
        decoded = self.decoder_norm(output)
        class_logits = self.class_embed(decoded)
        mask_logits = torch.einsum(
            "bqc,bchw->bqhw", self.mask_embed(decoded), mask_features
        )
        attention_mask = F.interpolate(
            mask_logits,
            size=attention_size,
            mode="bilinear",
            align_corners=False,
        )
        attention_mask = attention_mask.sigmoid().flatten(2) < 0.5
        attention_mask = attention_mask[:, None].expand(
            -1, self.num_heads, -1, -1
        ).flatten(0, 1).detach()
        return class_logits, mask_logits, attention_mask

    def forward(
        self, features: list[torch.Tensor], mask_features: torch.Tensor
    ):
        memories, positions, sizes = [], [], []
        for level, feature in enumerate(features):
            sizes.append(feature.shape[-2:])
            memories.append(
                feature.flatten(2).transpose(1, 2)
                + self.level_embed.weight[level][None, None]
            )
            positions.append(
                self.position(feature).flatten(2).transpose(1, 2).to(feature)
            )
        batch = features[0].shape[0]
        output = self.query_feature.weight[None].expand(batch, -1, -1)
        query_position = self.query_position.weight[None].expand(batch, -1, -1)
        class_logits, mask_logits, attention_mask = self.prediction_heads(
            output, mask_features, sizes[0]
        )
        for layer, (cross, self_attention, feed_forward) in enumerate(
            zip(self.cross_layers, self.self_layers, self.ffn_layers)
        ):
            level = layer % self.num_feature_levels
            fully_masked = attention_mask.all(dim=-1)
            if fully_masked.any():
                attention_mask = attention_mask.clone()
                attention_mask[fully_masked] = False
            output = cross(
                output,
                memories[level],
                query_position,
                positions[level],
                attention_mask,
            )
            output = self_attention(output, query_position)
            output = feed_forward(output)
            class_logits, mask_logits, attention_mask = self.prediction_heads(
                output,
                mask_features,
                sizes[(layer + 1) % self.num_feature_levels],
            )
        return class_logits, mask_logits


class XRaydarMask2Former(nn.Module):
    """Inference-only Q15 Mask2Former using its training-time EVA input."""

    def __init__(self):
        super().__init__()
        self.backbone = XRaydarInception3(num_classes=38, aux_logits=True)
        del self.backbone.fc
        del self.backbone.AuxLogits
        self.backbone.aux_logits = False
        self.pixel_decoder = XRaydarFPNPixelDecoder(hidden_dim=256)
        self.transformer_decoder = MultiScaleMaskedTransformerDecoder(
            hidden_dim=256,
            num_queries=15,
            heads=8,
            decoder_layers=9,
        )

    def features(self, image: torch.Tensor):
        # The shared training loader emitted EVA-standardized tensors. The
        # X-Raydar Mask2Former converted those tensors back to [0,1] and then
        # applied the X-Raydar normalization internally. Retaining these exact
        # operations avoids attention-level drift from a merely algebraic
        # simplification.
        x = image * 0.28509309 + 0.49185243
        x = (x - 0.491) / 0.271
        x = self.backbone._transform_input(x)
        x = self.backbone.Conv2d_1a_3x3(x)
        x = self.backbone.Conv2d_2a_3x3(x)
        x = self.backbone.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.backbone.Conv2d_3b_1x1(x)
        c2 = self.backbone.Conv2d_4a_3x3(x)
        x = F.max_pool2d(c2, kernel_size=3, stride=2)
        x = self.backbone.Mixed_5b(x)
        x = self.backbone.Mixed_5c(x)
        c3 = self.backbone.Mixed_5d(x)
        x = self.backbone.Mixed_6a(c3)
        x = self.backbone.Mixed_6b(x)
        x = self.backbone.Mixed_6c(x)
        x = self.backbone.Mixed_6d(x)
        c4 = self.backbone.Mixed_6e(x)
        x = self.backbone.Mixed_7a(c4)
        x = self.backbone.Mixed_7b(x)
        c5 = self.backbone.Mixed_7c(x)
        return [c2, c3, c4, c5]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mask_features, multi_scale = self.pixel_decoder(self.features(image))
        class_logits, mask_logits = self.transformer_decoder(
            multi_scale, mask_features
        )
        cavity_probability = class_logits.softmax(-1)[..., 0]
        instance_probability = (
            cavity_probability[..., None, None] * mask_logits.sigmoid()
        )
        probability = instance_probability.max(dim=1).values
        return F.interpolate(
            probability[:, None],
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).clamp_(1e-6, 1 - 1e-6)


def load_state(model: nn.Module, checkpoint_path: str) -> None:
    state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    message = model.load_state_dict(state, strict=True)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {message}")
