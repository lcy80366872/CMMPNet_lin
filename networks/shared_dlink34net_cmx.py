import torch.nn as nn
import torch

from .basic_blocks import *
from torchvision import models
from networks.attention_block import CBAMBlock,SEAttention
from networks.basic_blocks import Exchange,ModuleParallel,BatchNorm2dParallel

from networks.net_utils import *
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
        num_heads = [1, 2, 4, 8]
        #self.conv1 = ModuleParallel(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        #self.bn1 = BatchNorm2dParallel(64, num_parallel)
        #self.relu = ModuleParallel(nn.ReLU(inplace=True))
        #self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, blocks_num[0], bn_threshold)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], bn_threshold, stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], bn_threshold, stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], bn_threshold, stride=2)
        self.frm1 = FeatureRectifyModule(filters[0])
        self.frm2 = FeatureRectifyModule(filters[1])
        self.frm3 = FeatureRectifyModule(filters[2])
        self.frm4 = FeatureRectifyModule(filters[3])
        self.ffm1 = FeatureFusionModule(filters[0], num_heads=num_heads[0])
        self.ffm2 = FeatureFusionModule(filters[1], num_heads=num_heads[1])
        self.ffm3 = FeatureFusionModule(filters[2], num_heads=num_heads[2])
        self.ffm4 = FeatureFusionModule(filters[3], num_heads=num_heads[3])

        self.dblock = DBlock(filters[3])

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1)
        self.finalrelu2 = nonlinearity


        self.finalconv = nn.Conv2d(filters[0]//2, num_classes, 3, padding=1)
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
        x_1[0], x_1[1] = self.frm1(x_1[0], x_1[1])
        x_fused1 = self.ffm1(x_1[0], x_1[1])
        x_2 = self.layer2(x_1)
        x_2[0], x_2[1] = self.frm2(x_2[0], x_2[1])
        x_fused2 = self.ffm2(x_2[0], x_2[1])
        x_3 = self.layer3(x_2)
        x_3[0], x_3[1] = self.frm3(x_3[0], x_3[1])
        x_fused3 = self.ffm3(x_3[0], x_3[1])
        x_4 = self.layer4(x_3)
        x_4[0], x_4[1] = self.frm4(x_4[0], x_4[1])
        x_fused4 = self.ffm4(x_4[0], x_4[1])



        x_c = self.dblock(x_fused4)
        # decoder
        x_d4 = self.decoder4(x_c) + x_fused3
        x_d3 = self.decoder3(x_d4) + x_fused2
        x_d2 = self.decoder2(x_d3) + x_fused1
        x_d1 = self.decoder1(x_d2)


        x_out = self.finalrelu1(self.finaldeconv1(x_d1))
        x_out = self.finalrelu2(self.finalconv2(x_out))

        out = self.finalconv(x_out)
        out = torch.sigmoid(out)
        # out =nn.LogSoftmax()(ens)
        # out.append(ens)#[ä¸¤ä¸ªè¾“å…¥çš„outä»¥åŠä»–ä»¬æŒ‰alphaå‡è¡¡åŽçš„output,ä¸€å…±ä¸‰ä¸ª]

        return out


def DinkNet34_CMMPNet():
    model = ResNet(block=BasicBlock, blocks_num=[3, 4, 6, 3],num_parallel=2,num_classes=1,bn_threshold=0.02)
    return model
