import torch.nn as nn
import torch
from networks.CondConv import CondConv, DynamicConv
# from networks.deform_conv_v2 import DeformConv2d
from .basic_blocks import *
from torchvision import models
from networks.attention_block import CBAMBlock,SEAttention
from networks.basic_blocks import Exchange,ModuleParallel,BatchNorm2dParallel,DualGCN
from networks.Nonlocal import NLBlockND,NLBlockND_Fuse,CrissCrossAttention_Fuse
from networks.SGCN import TwofoldGCN
from networks.Freq import *
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
    def __init__(self, inplanes, planes, num_parallel, bn_threshold,stride=1, downsample=None,condconv=False):
        super(BasicBlock, self).__init__()
        self.condconv=condconv
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        if condconv == False:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 =nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
            # self.dynamicconv=DynamicConv(planes, planes, kernel_size=3, stride=1,padding=1, bias=False)
            self.deformconv=DeformConv2d(planes, planes,kernel_size=3,stride=1, padding=1, bias=None)
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
        out = self.bn1(out)
        out = self.relu(out)
        if self.condconv == False:
            out = self.conv2(out)
        else:
            out[0] = self.deformconv(out[0])
            out[1] = self.deformconv(out[1])
        out = self.bn2(out)
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)

        if self.downsample is not None:
            residual = self.downsample(x)

        # print('num_paraller', self.num_parallel)
        # print('lenout', len(out))
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

        self.exchange = Exchange_3()
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


        # self.conv1 = ModuleParallel(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
#         self.bn1 = BatchNorm2dParallel(64, num_parallel)
#         self.relu = ModuleParallel(nn.ReLU(inplace=True))
#         self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
#         self.bn_threshold=bn_threshold
#         self.exchange = Exchange()
#         self.bn_list = []
#         for module in self.bn1.modules():
#             if isinstance(module, nn.BatchNorm2d):
#                 self.bn_list.append(module)

        self.layer1 = self._make_layer(block, 64, blocks_num[0], bn_threshold, condconv=True)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], bn_threshold, stride=2, condconv=True)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], bn_threshold, stride=2, condconv=True)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], bn_threshold, stride=2, condconv=True)
        # self.gcn1=DualGCN(64)
        # self.gcn2 = DualGCN(128)
        # self.gcn3 = DualGCN(256)
        # self.gcn4 = DualGCN(512)
        # self.non_local2 = NLBlockND(filters[1], mode='embedded', dimension=2)
        # self.dropout = ModuleParallel(nn.Dropout(p=0.5))

        self.dblock = DBlock_parallel(filters[3],num_parallel)
        # self.SGCN=TwofoldGCN(filters[3] ,filters[3] ,filters[3]  )
        # self.dgcn_seg1 = TwofoldGCN(filters[0] ,filters[0] ,filters[0]  )
        # self.dgcn_seg2 = TwofoldGCN(filters[1] ,filters[1] ,filters[1]  )
        # self.dgcn_seg3 = TwofoldGCN(filters[2] ,filters[2] ,filters[2]  )
        # # decoder
        # self.decoder4 = DecoderBlock_parallel_exchange(filters[3], filters[2],num_parallel,bn_threshold)
        # self.decoder3 = DecoderBlock_parallel_exchange(filters[2], filters[1],num_parallel,bn_threshold)
        # self.decoder2 = DecoderBlock_parallel_exchange(filters[1], filters[0],num_parallel,bn_threshold)
        # self.decoder1 = DecoderBlock_parallel_exchange(filters[0], filters[0],num_parallel,bn_threshold)
        self.decoder4 = DecoderBlock_parallel(filters[3], filters[2], num_parallel)
        self.decoder3 = DecoderBlock_parallel(filters[2], filters[1], num_parallel)
        self.decoder2 = DecoderBlock_parallel(filters[1], filters[0], num_parallel)
        self.decoder1 = DecoderBlock_parallel(filters[0], filters[0], num_parallel)

        # self.dem_e1 = DEM(filters[0])
        # self.dem_e2 = DEM(filters[1])
        # self.dem_e3 = DEM(filters[2])
        # self.dem_e4 = DEM(filters[3])
        #
        # self.dem_d4 = DEM(filters[2])
        # self.dem_d3 = DEM(filters[1])
        # self.dem_d2 = DEM(filters[0])
        # self.dem_d1 = DEM(filters[0])

        self.finaldeconv1 = ModuleParallel(nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1))
        self.finalrelu1 =  ModuleParallel(nn.ReLU(inplace=True))
        # self.finalrelu1 = nonlinearity
        self.finalconv2 = ModuleParallel(nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1))
        self.finalrelu2 = ModuleParallel(nn.ReLU(inplace=True))
        self.se = SEAttention(filters[0] // 2, reduction=4)
        # self.se1 = SEAttention(filters[0] // 2, reduction=4)
        # self.atten=CBAMBlock(channel=filters[0], reduction=4, kernel_size=7)
        # self.fuse =NLBlockND_Fuse(filters[0]//2, filters[0]//2,mode='embedded', dimension=2)
        # self.fuse =CrissCrossAttention_Fuse(filters[0])
        self.finalconv = nn.Conv2d(filters[0], num_classes, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, num_blocks, bn_threshold, stride=1, condconv=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                BatchNorm2dParallel(planes * block.expansion, self.num_parallel)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, stride, downsample,condconv=condconv))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold,condconv=condconv))

        return nn.Sequential(*layers)

    def forward(self, inputs):


        x = inputs[:, :3, :, :]
        g = inputs[:, 3:4, :, :]

        # print('xxxxxxxx',ycbr.shape)
        # g =g.repeat([1,3,1,1])


        ##stem layer
        x = self.firstconv1(x)
        g = self.firstconv1_g(g)
        out = self.firstmaxpool(self.firstrelu(self.firstbn(x)))
        out_g = self.firstmaxpool_g(self.firstrelu_g(self.firstbn_g(g)))
        out=out,out_g

        # x=x,g
        # x = self.conv1(x)
        # x = self.bn1(x)
        # if len(x) > 1:
        #     x = self.exchange(x, self.bn_list, self.bn_threshold)
        # x = self.relu(x)
        # out = self.maxpool(x)

        ##layers:
        x_1 = self.layer1(out)
        # x_1 = self.dem_e1(x_1)
        x_2 = self.layer2(x_1)
        # x_2 = self.dem_e2(x_2)
        x_3 = self.layer3(x_2)
        # x_3 = self.dem_e3(x_3)
        x_4 = self.layer4(x_3)
        # x_4 = self.dem_e4(x_4)

        x_c = self.dblock(x_4)

       # decoder
        x_d4 = [self.decoder4(x_c)[l] + x_3[l] for l in range(self.num_parallel)]
        # x_d4 = self.dem_d4(x_d4)
        x_d3 = [self.decoder3(x_d4)[l] + x_2[l] for l in range(self.num_parallel)]
        # x_d3 = self.dem_d3(x_d3)
        x_d2 = [self.decoder2(x_d3)[l] + x_1[l] for l in range(self.num_parallel)]
        # x_d2 = self.dem_d2(x_d2)
        x_d1 = self.decoder1(x_d2)
        # x_d1 = self.dem_d1(x_d1)


        x_out = self.finalrelu1(self.finaldeconv1(x_d1))
        x_out = self.finalrelu2(self.finalconv2(x_out))

        x_out[0]=self.se(x_out[0])
        x_out[1] = self.se(x_out[1])
        out=self.finalconv(torch.cat((x_out[0], x_out[1]), 1))

        # out =self.finalconv(fuse)
        # out = self.finalconv(torch.cat((torch.cat((x_out[0], x_out[1]), 1),x_out[2]),1))
        # out=self.finalconv(x_out)
        # alpha_soft = F.softmax(self.alpha,dim=0)
        # ens = 0
        # for l in range(self.num_parallel):
        #     ens += alpha_soft[l] * out[l].detach()
        out = torch.sigmoid(out)
        # out =nn.LogSoftmax()(ens)
        # out.append(ens)#[濞戞挶鍊撻柌婊勬綇閹惧啿寮抽柣銊ュ煟ut濞寸姰鍎卞閿嬬閺嶏附绮﹂柟绋柯╨pha闁秆冩穿閵嗏偓闁告艾娴峰▓鎲僽tput,濞戞挴鍋撻柛蹇撳綁缁椾焦绋夐悮锟�

        return out#,freq_output_1,freq_output_2,freq_output_3


def DinkNet34_CMMPNet():
    model = ResNet(block=BasicBlock, blocks_num=[3, 4, 6, 3],num_parallel=2,num_classes=1,bn_threshold=2e-2)
    return model
