import torch.nn as nn
import torch
from networks.CondConv import CondConv, DynamicConv
from .basic_blocks import *
from torchvision import models
from networks.attention_block import CBAMBlock,SEAttention



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
        # self.sp_exchange=Spatial_Exchange()
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
        # if len(x) > 1:
        #     out = self.sp_exchange(out,0.6)
        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)

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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.conv1_g = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2,
                                 padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)#BatchNorm2dParallel(self.inplanes, num_parallel)
        self.bn1_g = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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

        self.layer1 = self._make_layer(block, 64, blocks_num[0], bn_threshold)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], bn_threshold, stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], bn_threshold, stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], bn_threshold, stride=2)

        # self.dropout = ModuleParallel(nn.Dropout(p=0.5))

        self.dblock = DBlock_parallel(filters[3],2)
        # self.dblock_add = DBlock(filters[3])
        # decoder
        self.decoder4 = DecoderBlock_parallel(filters[3], filters[2],2)
        self.decoder3 = DecoderBlock_parallel(filters[2], filters[1],2)
        self.decoder2 = DecoderBlock_parallel(filters[1], filters[0],2)
        self.decoder1 = DecoderBlock_parallel(filters[0], filters[0],2)
        # self.decoder4 = DecoderBlock_parallel_exchange(filters[3], filters[2],2,bn_threshold)
        # self.decoder3 = DecoderBlock_parallel_exchange(filters[2], filters[1],2,bn_threshold)
        # self.decoder2 = DecoderBlock_parallel_exchange(filters[1], filters[0],2,bn_threshold)
        # self.decoder1 = DecoderBlock_parallel_exchange(filters[0], filters[0],2,bn_threshold)


        # self.finaldeconv1_add = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        # self.finalrelu1_add = nonlinearity
        # self.finalconv2_add = nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1)
        # self.finalrelu2_add = nonlinearity

        self.finaldeconv1 = ModuleParallel(nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1))
        self.finalrelu1 =  ModuleParallel(nn.ReLU(inplace=True))
        #self.finalrelu1 = nonlinearity
        self.finalconv2 = ModuleParallel(nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1))
        self.finalrelu2 = ModuleParallel(nn.ReLU(inplace=True))
        self.se = SEAttention(filters[0] // 2, reduction=4)
        # self.atten=CBAMBlock(channel=filters[0], reduction=4, kernel_size=7)
        self.allgin=AlignModule(filters[0] // 2)
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

        ##stem layer
        x = self.firstconv1(x)
        g = self.firstconv1_g(g)
        out = self.firstmaxpool(self.firstrelu(self.firstbn(x)))
        out_g = self.firstmaxpool_g(self.firstrelu_g(self.firstbn_g(g)))

        out = out, out_g
        # out = torch.cat((out, out_g), 1)

        ##layers:
        x_1 = self.layer1(out)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        # x_4 =self.dropout(x_4)

        x_c = self.dblock(x_4)
        # decoder
        x_d4 = [self.decoder4(x_c)[l] + x_3[l] for l in range(self.num_parallel)]
        x_d3 = [self.decoder3(x_d4)[l] + x_2[l] for l in range(self.num_parallel)]
        x_d2 = [self.decoder2(x_d3)[l] + x_1[l] for l in range(self.num_parallel)]
        x_d1 = self.decoder1(x_d2)


        x_out = self.finalrelu1(self.finaldeconv1(x_d1))
        x_out = self.finalrelu2(self.finalconv2(x_out))

        x_out[0]=self.se(x_out[0])
        x_out[1] = self.se(x_out[1])
        # atten=self.atten(torch.cat((x_out[0], x_out[1]), 1))
        x_out =self.allgin(x_out)
        # out = self.finalconv(torch.cat((x_out[0], x_out[1]), 1))
        out=self.finalconv(x_out)
        # alpha_soft = F.softmax(self.alpha,dim=0)
        # ens = 0
        # for l in range(self.num_parallel):
        #     ens += alpha_soft[l] * out[l].detach()
        out = torch.sigmoid(out)
        # out =nn.LogSoftmax()(ens)
        # out.append(ens)#[Ã¥Â¨â€˜Ã¦â€™Â±Ã¥â‚¬â€œÃ©â€“Â²Ã¦Â»â€žÃ¦Â½ÂÃ©Å½Â¾Ã¥â€ Â²Ã¥Â¼Â³Ã©â€“Â»Ã£â€žÂ¥Ã¥Å¸Â£utÃ¥Â¨Â´Ã§Å Â®Ã¥â€žÂ±Ã¥Â¯Â®Ã©â€â€¹Ã§Â¦â€™Ã©ÂÂÃ¯Â¸Â½Ã§Â²Â¦Ã©â€“Â¹Ã§Â¨Â¿Ã‚Â¨lphaÃ©â€“Â¸Ã‘Æ’Ã¦Â´Â©Ã©Å â€ Ã¢â€šÂ¬Ã©â€“Â¸Ã¦Â°Â¬Ã¦ÂµÂ·Ã¥Â¨Ë†Ã¦â€ Æ’utput,Ã¥Â¨â€˜Ã¦â€™Â¯Ã¥Ââ€œÃ©â€“Â¸Ã¥Â¿â€œÃ¥Â½â€°Ã§Â»â€”Ã¤Â½Â¹Ã§Â¨â€°Ã©ÂÅ¡Ã¯Â¿Â½

        return out


def DinkNet34_CMMPNet():
    model = ResNet(block=BasicBlock, blocks_num=[3, 4, 6, 3],num_parallel=2,num_classes=1,bn_threshold=2e-2)
    return model
