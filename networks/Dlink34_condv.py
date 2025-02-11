import torch.nn as nn
import torch
from networks.CondConv import CondConv, DynamicConv
from .basic_blocks import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, downsample1=None, condconv=False, **kwargs):
        super(BasicBlock, self).__init__()
        self.condconv = condconv
        self.in_chanel = in_channel
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv1_g = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                 kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1_g = nn.BatchNorm2d(out_channel)

        if condconv == False:
            self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                   kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv2 =DynamicConv(out_channel, out_channel, kernel_size=3, stride=1,
                                  padding=1, bias=False)
        self.conv2_g = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.bn2_g = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.downsample1 = downsample1

    def forward(self, input):

        a = self.in_chanel
        x = input[:, :a, :, :]
        g = input[:, a:, :, :]
        # print('x:',x.shape)
        # print('g:', g.shape)
        residual = x
        residual_g = g
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        g = self.conv1_g(g)
        g = self.bn1_g(g)
        g = self.relu(g)
        # print('out:',out.shape)
        # print('g:', g.shape)
        if self.condconv == False:
            out = self.conv2(out)
        else:
            out = self.conv2(out, g)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        out_g = self.conv2_g(g)
        out_g = self.bn2_g(out_g)
        if self.downsample1 is not None:
            residual_g = self.downsample1(residual_g)
        out_g += residual_g
        out_g = self.relu(out_g)

        out = torch.cat((out, out_g), 1)

        return out

class BasicBlock0(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock0, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y

'''
Feature Separation Part
'''
class FSP(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FSP, self).__init__()
        self.filter = FilterLayer(2*in_planes, out_planes, reduction)

    def forward(self, guidePath, mainPath):

        combined = torch.cat((guidePath, mainPath), dim=1)
        channel_weight = self.filter(combined)
        out = mainPath + channel_weight * guidePath
        return out



class SE_OUT(torch.nn.Module):  # Dual Enhancement Module
    def __init__(self, in_planes, out_planes, reduction=16, bn_momentum=0.0003):
        self.init__ = super(SE_OUT, self).__init__()
        self.in_planes = in_planes
        self.bn_momentum = bn_momentum

        self.fsp_rgb = FSP(in_planes, out_planes, reduction)
        self.fsp_hha = FSP(in_planes, out_planes, reduction)

        self.gate_rgb = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)
        self.gate_hha = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x,y):
        rgb, hha = x,y
        b, c, h, w = rgb.size()

        rec_rgb = self.fsp_rgb(hha, rgb)
        rec_hha = self.fsp_hha(rgb, hha)

        cat_fea = torch.cat([rec_rgb, rec_hha], dim=1)

        attention_vector_l = self.gate_rgb(cat_fea)
        attention_vector_r = self.gate_hha(cat_fea)

        attention_vector = torch.cat([attention_vector_l, attention_vector_r], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        merge_feature = rgb*attention_vector_l + hha*attention_vector_r
        merge = self.relu1(merge_feature)

        return merge

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,

                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        filters = [64, 128, 256, 512]

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.conv1_g = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,
                                 padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.bn1_g = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0], condconv=True)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2, condconv=True)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2, condconv=True)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2, condconv=True)

        # self.layer1_add = self._make_layer(block, 64, blocks_num[0], condconv=False)
        # self.layer2_add = self._make_layer(block, 128, blocks_num[1], stride=2, condconv=False)
        # self.layer3_add = self._make_layer(block, 256, blocks_num[2], stride=2, condconv=False)
        # self.layer4_add = self._make_layer(block, 512, blocks_num[3], stride=2, condconv=False)

        self.dblock = DBlock(filters[3])
        # self.OUT_SE = SE_OUT(filters[3], filters[3])
        self.dblock_add = DBlock(filters[3])
        # decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.decoder4_add = DecoderBlock(filters[3], filters[2])
        self.decoder3_add = DecoderBlock(filters[2], filters[1])
        self.decoder2_add = DecoderBlock(filters[1], filters[0])
        self.decoder1_add = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1_add = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1_add = nonlinearity
        self.finalconv2_add = nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1)
        self.finalrelu2_add = nonlinearity

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv = nn.Conv2d(filters[0], num_classes, 3, padding=1)
        # self.finalconv = nn.Conv2d(filters[0] // 2, num_classes, 3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1, condconv=False):
        downsample = None
        downsample1 = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
            downsample1 = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            downsample1=downsample1,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group,
                            condconv=condconv
                            ))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group,
                                condconv=condconv
                                ))

        return nn.Sequential(*layers)

    def forward(self, inputs):

        x = inputs[:, :3, :, :]
        g = inputs[:, 3:, :, :]

        ##stem layer
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out_g = self.relu(self.bn1_g(self.conv1_g(g)))
        out_g = self.maxpool(out_g)

        # out = out, out_g
        out = torch.cat((out, out_g), 1)

        ##layers:
        x_1 = self.layer1(out)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        x_e1 = x_1[:, :64, :, :]
        g_e1 = x_1[:, 64:, :, :]

        x_e2 = x_2[:, :128, :, :]
        g_e2 = x_2[:, 128:, :, :]

        x_e3 = x_3[:, :256, :, :]
        g_e3 = x_3[:, 256:, :, :]

        x_e4 = x_4[:, :512, :, :]
        g_e4 = x_4[:, 512:, :, :]

        g_c = self.dblock(g_e4)
        x_c = self.dblock(x_e4)
        # decoder
        x_d4 = self.decoder4(x_c) + x_e3
        x_d3 = self.decoder3(x_d4) + x_e2
        x_d2 = self.decoder2(x_d3) + x_e1
        x_d1 = self.decoder1(x_d2)

        g_d4 = self.decoder4_add(g_c) + g_e3
        g_d3 = self.decoder3_add(g_d4) + g_e2
        g_d2 = self.decoder2_add(g_d3) + g_e1
        g_d1 = self.decoder1_add(g_d2)

        x_out = self.finalrelu1(self.finaldeconv1(x_d1))
        x_out = self.finalrelu2(self.finalconv2(x_out))
        g_out = self.finalrelu1(self.finaldeconv1(g_d1))
        g_out = self.finalrelu2(self.finalconv2(g_out))
        out = self.finalconv(torch.cat((x_out, g_out), 1))
        # out=self.finalconv(x_out)
        out = torch.sigmoid(out)

        return out


def DinkNet34_CMMPNet():
    model = ResNet(BasicBlock, [3, 4, 6, 3], 1)
    return model


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
