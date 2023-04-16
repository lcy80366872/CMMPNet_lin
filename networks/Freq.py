import torch
import torch.nn as nn
from einops import rearrange
from torch import nn, einsum
from abc import ABC
import torch_dct as DCT



class PreNorm(nn.Module, ABC):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module, ABC):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module, ABC):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

def norm(x):
    return (1 - torch.exp(-x)) / (1 + torch.exp(-x))


def norm_(x):
    import numpy as np
    return (1 - np.exp(-x)) / (1 + np.exp(-x))


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class HOR(nn.Module):
    def __init__(self):
        super(HOR, self).__init__()
        self.high = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.low = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.value = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        self.e_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.latter = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

    def forward(self, x_latter, x):
        b, c, h, w = x_latter.shape
        _, c_, _, _ = x.shape
        x_latter_ = self.high(x_latter).reshape(b, c, h * w).contiguous()
        x_ = self.low(x).reshape(b, c_, h * w).permute(0, 2, 1).contiguous()

        p = torch.bmm(x_, x_latter_).contiguous()
        p = self.softmax(p).contiguous()

        e_ = torch.bmm(p, self.value(x).reshape(b, c, h * w).permute(0, 2, 1)).contiguous()
        e = e_ + x_
        e = e.permute(0, 2, 1).contiguous()
        e = self.e_conv(e.reshape(b, c, h, w)).reshape(b, c, h * w).contiguous()

        # e = e.permute(0, 2, 1)
        x_latter_ = self.latter(x_latter).reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        t = torch.bmm(e, x_latter_).contiguous()
        t = self.softmax(t).contiguous()

        x_ = self.mid(x).view(b, c_, h * w).permute(0, 2, 1).contiguous()
        out = torch.bmm(x_, t).permute(0, 2, 1).reshape(b, c, h, w).contiguous()

        return out


class channel_shuffle(nn.Module):
    def __init__(self,groups=4):
        super(channel_shuffle,self).__init__()
        self.groups=groups
    def forward(self,x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups,
               channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

class two_ConvBnRule(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(two_ConvBnRule, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, mid=False):
        feat = self.conv1(x)
        feat = self.BN1(feat)
        feat = self.relu1(feat)

        if mid:
            feat_mid = feat

        feat = self.conv2(feat)
        feat = self.BN2(feat)
        feat = self.relu2(feat)

        if mid:
            return feat, feat_mid
        return feat

def Seg():

    dict = {0: 0, 1: 1, 2: 8, 3: 16, 4: 9, 5: 2, 6: 3, 7: 10, 8: 17,
                 9: 24, 10: 32, 11: 25, 12: 18, 13: 11, 14: 4, 15: 5, 16: 12,
                 17: 19, 18: 26, 19: 33, 20: 40, 21: 48, 22: 41, 23: 34, 24: 27,
                 25: 20, 26: 13, 27: 6, 28: 7, 29: 14, 30: 21, 31: 28, 32: 35,
                 33: 42, 34: 49, 35: 56, 36: 57, 37: 50, 38: 43, 39: 36, 40: 29,
                 41: 22, 42: 15, 43: 23, 44: 30, 45: 37, 46: 44, 47: 51, 48: 58,
                 49: 59, 50: 52, 51: 45, 52: 38, 53: 31, 54: 39, 55: 46, 56: 53,
                 57: 60, 58: 61, 59: 54, 60: 47, 61: 55, 62: 62, 63: 63}
    a = torch.zeros(1, 64, 1, 1)

    for i in range(0, 32):
        a[0, dict[i+32], 0, 0] = 1

    return a


class PAM(nn.Module):

    def __init__(self, in_dim):

        super(PAM, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim*2,
            out_channels= 2,
            kernel_size=3,
            padding=1
        )

        self.v_rgb = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)
        self.v_freq = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)

    def forward(self, rgb, freq):

        attmap = self.conv( torch.cat( (rgb,freq),1) )
        attmap = torch.sigmoid(attmap)

        rgb = attmap[:,0:1,:,:] * rgb * self.v_rgb
        freq = attmap[:,1:,:,:] * freq * self.v_freq
        out = rgb + freq

        return out

def DCT_Operation(x):
    num_batchsize = x.shape[0]
    size = x.shape[2]

    ycbcr_image = x.reshape(num_batchsize, 4, size // 8, 8, size // 8, 8).permute(0, 2, 4, 1, 3, 5)
    ycbcr_image = DCT.dct_2d(ycbcr_image, norm='ortho')
    ycbcr_image = ycbcr_image.reshape(num_batchsize, size // 8, size // 8, -1).permute(0, 3, 1, 2)
    return ycbcr_image

class FEM(nn.Module):
    def __init__(self,in_chanel=192):
        super(FEM, self).__init__()
        self.seg=Seg()
        self.shuffle = channel_shuffle()

        self.high_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)
        self.low_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)

        self.band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)
        self.spatial = Transformer(dim=192, depth=1, heads=2, dim_head=64, mlp_dim=64 * 2, dropout=0)

        self.con1_2 = nn.Conv2d(in_channels=in_chanel, out_channels=64, kernel_size=1)
        self.con1_3 = nn.Conv2d(in_channels=in_chanel, out_channels=64, kernel_size=1)
        self.con1_4 = nn.Conv2d(in_channels=in_chanel, out_channels=64, kernel_size=1)
        self.con1_5 = nn.Conv2d(in_channels=in_chanel, out_channels=64, kernel_size=1)
        self.vector_y = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
        self.vector_cb = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
        self.vector_cr = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)

        self.freq_out_1 = nn.Conv2d(64, 1, 1, 1, 0)
        self.freq_out_2 = nn.Conv2d(64, 1, 1, 1, 0)
        self.freq_out_3 = nn.Conv2d(64, 1, 1, 1, 0)
        self.freq_out_4 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, DCT_x):
        # DCT_x =DCT_x.cuda()
        self.seg = self.seg.to(DCT_x.device)
        # print('device:',DCT_x.device)
        # print('device_vect:',self.vector_y.device)
        feat_y = DCT_x[:, 0:64, :, :] * (self.seg + norm(self.vector_y))
        feat_Cb = DCT_x[:, 64:128, :, :] * (self.seg + norm(self.vector_cb))
        feat_Cr = DCT_x[:, 128:192, :, :] * (self.seg + norm(self.vector_cr))

        origin_feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)
        origin_feat_DCT = self.shuffle(origin_feat_DCT)

        high = torch.cat([feat_y[:, 32:, :, :], feat_Cb[:, 32:, :, :], feat_Cr[:, 32:, :, :]], 1)
        low = torch.cat([feat_y[:, :32, :, :], feat_Cb[:, :32, :, :], feat_Cr[:, :32, :, :]], 1)

        b, n, h, w = high.shape
        high = torch.nn.functional.interpolate(high, size=(16, 16))
        low = torch.nn.functional.interpolate(low, size=(16, 16))

        high = rearrange(high, 'b n h w -> b n (h w)')
        low = rearrange(low, 'b n h w -> b n (h w)')

        high = self.high_band(high)
        low = self.low_band(low)

        y_h, b_h, r_h = torch.split(high, 32, 1)
        y_l, b_l, r_l = torch.split(low, 32, 1)
        feat_y = torch.cat([y_l, y_h], 1)
        feat_Cb = torch.cat([b_l, b_h], 1)
        feat_Cr = torch.cat([r_l, r_h], 1)

        feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)

        feat_DCT = self.band(feat_DCT)
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = self.spatial(feat_DCT)
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = rearrange(feat_DCT, 'b n (h w) -> b n h w', h=16)
        feat_DCT = torch.nn.functional.interpolate(feat_DCT, size=(h, w))

        feat_DCT = origin_feat_DCT + feat_DCT
        return feat_DCT
class FEM_gps(nn.Module):
    def __init__(self,in_chanel=64):
        super(FEM_gps, self).__init__()
        self.seg=Seg()
        self.shuffle = channel_shuffle()

        self.high_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)
        self.low_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)

        self.band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)
        self.spatial = Transformer(dim=64, depth=1, heads=2, dim_head=64, mlp_dim=64 * 2, dropout=0)

        self.con1_2 = nn.Conv2d(in_channels=in_chanel, out_channels=64, kernel_size=1)
        self.con1_3 = nn.Conv2d(in_channels=in_chanel, out_channels=64, kernel_size=1)
        self.con1_4 = nn.Conv2d(in_channels=in_chanel, out_channels=64, kernel_size=1)
        self.con1_5 = nn.Conv2d(in_channels=in_chanel, out_channels=64, kernel_size=1)
        self.vector_y = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
        self.vector_cb = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
        self.vector_cr = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)

        self.freq_out_1 = nn.Conv2d(64, 1, 1, 1, 0)
        self.freq_out_2 = nn.Conv2d(64, 1, 1, 1, 0)
        self.freq_out_3 = nn.Conv2d(64, 1, 1, 1, 0)
        self.freq_out_4 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, DCT_x):
        # DCT_x =DCT_x.cuda()
        self.seg = self.seg.to(DCT_x.device)
        # print('device:',DCT_x.device)
        # print('device_vect:',self.vector_y.device)
        feat = DCT_x[:, 0:64, :, :] * (self.seg + norm(self.vector_y))
        origin_feat_DCT = self.shuffle(feat)

        high = feat[:, 32:, :, :]
        low = feat[:, :32, :, :]

        b, n, h, w = high.shape
        high = torch.nn.functional.interpolate(high, size=(16, 16))
        low = torch.nn.functional.interpolate(low, size=(16, 16))

        high = rearrange(high, 'b n h w -> b n (h w)')
        low = rearrange(low, 'b n h w -> b n (h w)')

        high = self.high_band(high)
        low = self.low_band(low)

        feat = torch.cat([high,low], 1)

        feat_DCT = self.band(feat)
        feat_DCT = feat_DCT.transpose(1, 2)
        # print('ssss',feat_DCT.shape)
        feat_DCT = self.spatial(feat_DCT)
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = rearrange(feat_DCT, 'b n (h w) -> b n h w', h=16)
        feat_DCT = torch.nn.functional.interpolate(feat_DCT, size=(h, w))

        feat_DCT = origin_feat_DCT + feat_DCT
        return feat_DCT
