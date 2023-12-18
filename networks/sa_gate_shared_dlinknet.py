import torch.nn as nn
import torch

from .basic_blocks import *
from torchvision import models

from networks.basic_blocks import Exchange,ModuleParallel,BatchNorm2dParallel

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


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
        self.filter = FilterLayer(2 * in_planes, out_planes, reduction)

    def forward(self, guidePath, mainPath):
        combined = torch.cat((guidePath, mainPath), dim=1)
        channel_weight = self.filter(combined)
        out = mainPath + channel_weight * guidePath
        return out


class DEM(torch.nn.Module):  # Dual Enhancement Module
    def __init__(self, in_planes, out_planes, reduction=16, bn_momentum=0.0003):
        self.init__ = super(DEM, self).__init__()
        self.in_planes = in_planes
        self.bn_momentum = bn_momentum

        self.fsp_rgb = FSP(in_planes, out_planes, reduction)
        self.fsp_hha = FSP(in_planes, out_planes, reduction)

        self.gate_rgb = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.gate_hha = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        rgb, hha = x, y
        b, c, h, w = rgb.size()

        rec_rgb = self.fsp_rgb(hha, rgb)
        rec_hha = self.fsp_hha(rgb, hha)

        cat_fea = torch.cat([rec_rgb, rec_hha], dim=1)

        attention_vector_l = self.gate_rgb(cat_fea)
        attention_vector_r = self.gate_hha(cat_fea)

        attention_vector = torch.cat([attention_vector_l, attention_vector_r], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        merge_feature = rgb * attention_vector_l + hha * attention_vector_r

        rgb_out = (rgb + merge_feature) / 2
        hha_out = (hha + merge_feature) / 2

        rgb_out = self.relu1(rgb_out)
        hha_out = self.relu2(hha_out)

        return rgb_out, hha_out


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=1, bias=bias))


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, num_parallel, bn_threshold,stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # print('conv1',out[1].shape)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # if len(x) > 1:
        #     out = self.exchange(out, self.bn2_list, self.bn_threshold)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = BatchNorm2dParallel(planes * 4, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)


    def forward(self, x):
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_parallel=2,
                 num_classes=1,
                 bn_threshold=2e-2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.num_parallel=num_parallel

        filters = [64, 128, 256, 512]
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
        #                        padding=3, bias=False)
        # self.conv1_g = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2,
        #                          padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)#BatchNorm2dParallel(self.inplanes, num_parallel)
        # self.bn1_g = nn.BatchNorm2d(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        resnet = models.resnet34(pretrained=True)
        self.firstconv1 = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        resnet1 = models.resnet34(pretrained=True)
        self.firstconv1_g = nn.Conv2d(1, filters[0], kernel_size=7, stride=2, padding=3)
        self.firstbn_g = resnet1.bn1
        self.firstrelu_g = resnet1.relu
        self.firstmaxpool_g = resnet1.maxpool

        #self.conv1 = ModuleParallel(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        #self.bn1 = BatchNorm2dParallel(64, num_parallel)
        #self.relu = ModuleParallel(nn.ReLU(inplace=True))
        #self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, blocks_num[0], bn_threshold)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], bn_threshold, stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], bn_threshold, stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], bn_threshold, stride=2)
        # self.non_local2 = NLBlockND(filters[1], mode='embedded', dimension=2)
        # self.dropout = ModuleParallel(nn.Dropout(p=0.5))

        self.dblock = DBlock_parallel(filters[3],2)

        self.decoder4 = DecoderBlock_parallel(filters[3], filters[2], 2)
        self.decoder3 = DecoderBlock_parallel(filters[2], filters[1], 2)
        self.decoder2 = DecoderBlock_parallel(filters[1], filters[0], 2)
        self.decoder1 = DecoderBlock_parallel(filters[0], filters[0], 2)
        self.dem_e1 = DEM(filters[0], filters[0])
        self.dem_e2 = DEM(filters[1], filters[1])
        self.dem_e3 = DEM(filters[2], filters[2])
        self.dem_e4 = DEM(filters[3], filters[3])

        self.dem_d4 = DEM(filters[2], filters[2])
        self.dem_d3 = DEM(filters[1], filters[1])
        self.dem_d2 = DEM(filters[0], filters[0])
        self.dem_d1 = DEM(filters[0], filters[0])

        self.finaldeconv1 = ModuleParallel(nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1))
        self.finalrelu1 =  ModuleParallel(nn.ReLU(inplace=True))
        #self.finalrelu1 = nonlinearity
        self.finalconv2 = ModuleParallel(nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1))
        self.finalrelu2 = ModuleParallel(nn.ReLU(inplace=True))
        #self.se = SEAttention(filters[0] // 2, reduction=4)
        # self.se1 = SEAttention(filters[0] // 2, reduction=4)
        # self.atten=CBAMBlock(channel=filters[0], reduction=4, kernel_size=7)
        self.finalconv = nn.Conv2d(filters[0], num_classes, 3, padding=1)
        # self.finalconv = ModuleParallel(nn.Conv2d(filters[0] // 2, num_classes, 3, padding=1))
        # self.alpha = nn.Parameter(torch.ones(num_parallel, requires_grad=True))
        # self.register_parameter('alpha', self.alpha)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, num_blocks, bn_threshold, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                BatchNorm2dParallel(planes * block.expansion, self.num_parallel)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold))

        return nn.Sequential(*layers)

    def forward(self, inputs):

        x = inputs[:, :3, :, :]
        g = inputs[:, 3:, :, :]
        #g =g.repeat([1,3,1,1])#è½¬åŒ–ä¸ºä¸‰é€šé“

        ##stem layer
        x = self.firstconv1(x)
        g = self.firstconv1_g(g)
        out = self.firstmaxpool(self.firstrelu(self.firstbn(x)))
        out_g = self.firstmaxpool_g(self.firstrelu_g(self.firstbn_g(g)))

        #x=x,g
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        #out = self.maxpool(x)

        out = out, out_g

        ##layers:
        x_1 = self.layer1(out)
        x_1[0], x_1[1] = self.dem_e1(x_1[0], x_1[1])
        x_2 = self.layer2(x_1)
        x_2[0], x_2[1] = self.dem_e2(x_2[0], x_2[1])
        x_3 = self.layer3(x_2)
        x_3[0], x_3[1] = self.dem_e3(x_3[0], x_3[1])
        x_4 = self.layer4(x_3)
        x_4[0], x_4[1] = self.dem_e4(x_4[0], x_4[1])

        # x_4 =self.dropout(x_4)

        x_c = self.dblock(x_4)
        # decoder
        x_d4 = [self.decoder4(x_c)[l] + x_3[l] for l in range(self.num_parallel)]
        x_d4[0], x_d4[1] = self.dem_d4(x_d4[0], x_d4[1])
        x_d3 = [self.decoder3(x_d4)[l] + x_2[l] for l in range(self.num_parallel)]
        x_d3[0], x_d3[1] = self.dem_d3(x_d3[0], x_d3[1])
        x_d2 = [self.decoder2(x_d3)[l] + x_1[l] for l in range(self.num_parallel)]
        x_d2[0], x_d2[1] = self.dem_d2(x_d2[0], x_d2[1])
        x_d1 = self.decoder1(x_d2)
        x_d1[0], x_d1[1] = self.dem_d1(x_d1[0], x_d1[1])

        x_out = self.finalrelu1(self.finaldeconv1(x_d1))
        x_out = self.finalrelu2(self.finalconv2(x_out))


        out = self.finalconv(torch.cat((x_out[0], x_out[1]), 1))
        # out=self.finalconv(x_out)
        # alpha_soft = F.softmax(self.alpha,dim=0)
        # ens = 0
        # for l in range(self.num_parallel):
        #     ens += alpha_soft[l] * out[l].detach()
        out = torch.sigmoid(out)
        # out =nn.LogSoftmax()(ens)
        # out.append(ens)#[ä¸¤ä¸ªè¾“å…¥çš„outä»¥åŠä»–ä»¬æŒ‰alphaå‡è¡¡åŽçš„output,ä¸€å…±ä¸‰ä¸ª]

        return out


def DinkNet34_CMMPNet():
    model = ResNet(block=BasicBlock, blocks_num=[3, 4, 6, 3],num_parallel=2,num_classes=1,bn_threshold=0.02)
    return model
