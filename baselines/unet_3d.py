import torch
from torch import nn
from utils import *

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, pad=1, dropout=False):
        super().__init__()
        layers = [
            nn.Conv3d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Conv3d(out_c, out_c, kernel_size=kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class UpMerge(nn.Module): # Deconvolution
    def __init__(self, in_c, out_c, kernel=3, pad=1, last=False):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )
        self.conv = ConvBlock(out_c*2, out_c, kernel=kernel, pad=pad)

    def forward(self, x, skip_x):
        x = self.deconv(x)
        if x.shape[2:] != skip_x.shape[2:]:
            x = match_size_3D(x, skip_x.shape[2:])
        x = torch.cat((x, skip_x), dim=1)
        x = self.conv(x)
        return x


class SegEncoder(nn.Module):
    def __init__(self, args, in_c=1):
        super().__init__()
        self.args = args
        nf = self.args.nf
        self.pooling = nn.MaxPool3d(kernel_size=2)
        self.conv1 = ConvBlock(in_c, nf)
        self.conv2 = ConvBlock(nf, nf*2)
        self.conv3 = ConvBlock(nf*2, nf*4)
        self.conv4 = ConvBlock(nf*4, nf*8)
        self.conv5 = ConvBlock(nf*8, nf*16)
        self.conv = ConvBlock(nf*16, nf*16)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pooling(c1)
        c2 = self.conv2(p1)
        p2 = self.pooling(c2)
        c3 = self.conv3(p2)
        p3 = self.pooling(c3)
        c4 = self.conv4(p3)
        p4 = self.pooling(c4)
        c5 = self.conv5(p4)
        c5 = self.conv(c5)
        enc_list = [c1, c2, c3, c4, c5]
        return enc_list


class SegDecoder(nn.Module):
    def __init__(self, args, out_c=4):
        super().__init__()
        self.args = args
        nf = self.args.nf
        self.up1 = UpMerge(nf*16, nf*8)
        self.up2 = UpMerge(nf*8, nf*4)
        self.up3 = UpMerge(nf*4, nf*2)
        self.up4 = UpMerge(nf*2, nf)
        self.out = nn.Sequential(
            nn.Conv3d(nf, out_c, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, enc_list):
        c1, c2, c3, c4, c5 = enc_list
        u1 = self.up1(c5, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)
        out = self.out(u4)
        return out
    
