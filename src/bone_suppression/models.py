"""Bone-suppression networks.

All models are fully convolutional and, by default, *residual*: the network
predicts the bone layer B and the output is  y = x - B.  Learning the
(sparse) bone layer is easier than re-synthesising the whole image.

- unet       : U-Net [Ronneberger 2015] with GroupNorm, any input size.
- resnet_bs  : ResNet-BS [Rajaraman 2021], 16 residual blocks at full resolution.
- ae         : convolutional autoencoder, in the spirit of Gusarev et al. 2017.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), _norm(cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), _norm(cout), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32, depth: int = 4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.depth = depth
        self.inc = DoubleConv(in_ch, chs[0])
        self.downs = nn.ModuleList(DoubleConv(chs[i], chs[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(DoubleConv(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth)))
        self.outc = nn.Conv2d(chs[0], out_ch, 1)
        self.last_conv = self.outc

    def forward(self, x):
        h, w = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (m - h % m) % m, (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(F.max_pool2d(skips[-1], 2)))
        z = skips.pop()
        for up in self.ups:
            skip = skips.pop()
            z = F.interpolate(z, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            z = up(torch.cat([z, skip], dim=1))
        return self.outc(z)[..., :h, :w]


class ResBlock(nn.Module):
    def __init__(self, ch: int, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(ch, ch, 3, padding=1)
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.body(x)


class ResNetBS(nn.Module):
    """Full-resolution residual network (no down-sampling) as in ResNet-BS.

    Memory heavy at 512x512: train it on random crops (--crop 256).
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, ch: int = 64, n_blocks: int = 16):
        super().__init__()
        self.head = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(n_blocks)],
                                  nn.Conv2d(ch, ch, 3, padding=1))
        self.tail = nn.Conv2d(ch, out_ch, 3, padding=1)
        self.last_conv = self.tail

    def forward(self, x):
        f = self.head(x)
        return self.tail(f + self.body(f))


class AutoEncoderBS(nn.Module):
    """Convolutional encoder-decoder without skip connections (Gusarev-style AE)."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4

        def enc(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), _norm(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1), _norm(cout), nn.ReLU(inplace=True))

        def dec(cin, cout):
            return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1), _norm(cout),
                                 nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(enc(in_ch, c1), enc(c1, c2), enc(c2, c3))
        self.decoder = nn.Sequential(dec(c3, c2), dec(c2, c1), dec(c1, c1))
        self.out = nn.Conv2d(c1, out_ch, 3, padding=1)
        self.last_conv = self.out

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return self.out(self.decoder(self.encoder(x)))[..., :h, :w]


class BoneSuppressor(nn.Module):
    """Wraps a backbone. residual=True: y = x - backbone(x) (backbone predicts bones)."""

    def __init__(self, backbone: nn.Module, residual: bool = True):
        super().__init__()
        self.backbone = backbone
        self.residual = residual
        if residual and hasattr(backbone, "last_conv"):
            # zero-init: the untrained model is the identity (output = input),
            # so training starts from the "no suppression" baseline.
            nn.init.zeros_(backbone.last_conv.weight)
            nn.init.zeros_(backbone.last_conv.bias)

    def forward(self, x):
        out = self.backbone(x)
        return x - out if self.residual else out


MODELS = {"unet": UNet, "resnet_bs": ResNetBS, "ae": AutoEncoderBS}


def build_model(name: str, residual: bool = True, **kwargs) -> BoneSuppressor:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    return BoneSuppressor(MODELS[name](**kwargs), residual=residual)


def load_checkpoint(path, device="cpu") -> tuple[BoneSuppressor, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], residual=ckpt.get("residual", True), **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
