import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn import functional as F

#MPM模块
class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0]) #池化后的大小为20*20*c
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])   #池化后的大小为12*12*c
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))  #横向条形池化
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))  #纵向条形池化

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        #卷积核1*3，padding对应（0，1）来使卷积前后大小不变，实际上就是一维卷积，因为2_3和2_4都是对条形池化后的输出来进行卷积的
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        #2_4和2_5是条形池化+一维卷积
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)

class SPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(SPHead, self).__init__()
        inter_channels = in_channels // 2
        self.trans_layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
                norm_layer(inter_channels),
                nn.ReLU(True)
        )
        self.strip_pool1 = StripPooling(inter_channels, (4, 2), norm_layer, up_kwargs)
        self.strip_pool2 = StripPooling(inter_channels, (4, 2), norm_layer, up_kwargs)
        self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
                norm_layer(inter_channels // 2),
                nn.ReLU(True),
                nn.Dropout2d(0.1, False),
                nn.Conv2d(inter_channels // 2, out_channels, 1))

    def forward(self, x):
        x = self.trans_layer(x)
        x = self.strip_pool1(x)
        x = self.strip_pool2(x)
        x = self.score_layer(x)
        return x
