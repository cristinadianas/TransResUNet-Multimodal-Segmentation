import torch
import torch.nn as nn
from intermediate_fusion.resnet import resnet50
import numpy as np
import cv2


def save_feats_mean(x):
    b, c, h, w = x.shape
    if h == 256:
        with torch.no_grad():
            x = x.detach().cpu().numpy()
            x = np.transpose(x[0], (1, 2, 0))
            x = np.mean(x, axis=-1)
            x = x/np.max(x)
            x = x * 255.0
            x = x.astype(np.uint8)
            x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
            x = np.array(x, dtype=np.uint8)
            return x


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        backbone = resnet50(pretrained=pretrained, in_channels=1)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x: torch.Tensor):
        s1 = self.layer0(x)
        s2 = self.layer1(s1)
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        return s1, s2, s3, s4


class Bridge(nn.Module):
    def __init__(self, in_c: int = 1024, bottleneck_c: int = 256, num_layers: int = 2):
        super().__init__()
        self.b1 = Bottleneck(in_c, bottleneck_c, num_layers=num_layers)
        self.b2 = DilatedConv(in_c, bottleneck_c)

    def forward(self, s4: torch.Tensor):
        b1 = self.b1(s4)
        b2 = self.b2(s4)
        b3 = torch.cat([b1, b2], dim=1)
        return b3


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, num_layers=2, nhead=8):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_c,
            nhead=nhead,
            batch_first=True
        )
        self.tblock = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        b, c, h, w = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.tblock(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)

        x = self.conv2(x)
        return x


class DilatedConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=9, dilation=9),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c5 = nn.Sequential(
            nn.Conv2d(out_c*4, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.c5(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c[0]+in_c[1], out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r1(x)
        x = self.r2(x)
        return x


class TResUnet(nn.Module):
    def __init__(self, in_channels: int = 4, pretrained: bool = False):
        super().__init__()

        self.in_channels = in_channels
        self.num_modalities = in_channels

        if self.num_modalities < 1:
            raise ValueError(f"in_channels must be >= 1. Got {in_channels}.")

        self.encoders = nn.ModuleList([
            EncoderBlock(pretrained=pretrained)
            for _ in range(self.num_modalities)
        ])

        def adapter(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.adapt_s0 = adapter(self.num_modalities, self.num_modalities)

        self.adapt_s1 = adapter(self.num_modalities * 64, 64)
        self.adapt_s2 = adapter(self.num_modalities * 256, 256)
        self.adapt_s3 = adapter(self.num_modalities * 512, 512)
        self.adapt_s4 = adapter(self.num_modalities * 1024, 1024)

        self.bridge = Bridge(in_c=1024, bottleneck_c=256, num_layers=2)

        self.d1 = DecoderBlock([512, 512], 256)
        self.d2 = DecoderBlock([256, 256], 128)
        self.d3 = DecoderBlock([128, 64], 64)
        self.d4 = DecoderBlock([64, self.num_modalities], 32)

        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, heatmap=None, return_features=False):
        if x.ndim != 4:
            raise ValueError(f"Expected x to be 4D [B,C,H,W], got shape={tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected x to have {self.num_modalities} channels, got {x.shape[1]}")

        feats = {} if return_features else None

        s0 = self.adapt_s0(x)
        if feats is not None:
            feats["s0"] = s0

        s1_list, s2_list, s3_list, s4_list = [], [], [], []
        for m, enc in enumerate(self.encoders):
            xm = x[:, m:m+1, :, :]
            s1m, s2m, s3m, s4m = enc(xm)

            s1_list.append(s1m)
            s2_list.append(s2m)
            s3_list.append(s3m)
            s4_list.append(s4m)

            if feats is not None:
                feats[f"s1_m{m}"] = s1m
                feats[f"s2_m{m}"] = s2m
                feats[f"s3_m{m}"] = s3m
                feats[f"s4_m{m}"] = s4m

        s1 = self.adapt_s1(torch.cat(s1_list, dim=1))
        s2 = self.adapt_s2(torch.cat(s2_list, dim=1))
        s3 = self.adapt_s3(torch.cat(s3_list, dim=1))
        s4 = self.adapt_s4(torch.cat(s4_list, dim=1))

        if feats is not None:
            feats["s1"] = s1
            feats["s2"] = s2
            feats["s3"] = s3
            feats["s4"] = s4

        b3 = self.bridge(s4)
        if feats is not None:
            feats["b3"] = b3

        d1 = self.d1(b3, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        d4 = self.d4(d3, s0)

        if feats is not None:
            feats["d1"] = d1
            feats["d2"] = d2
            feats["d3"] = d3
            feats["d4"] = d4

        y = self.output(d4)

        if heatmap != None:
            hmap = save_feats_mean(d4)
            if return_features:
                return hmap, y, feats
            return hmap, y

        if return_features:
            return y, feats
        return y