import torch.nn as nn
import torch
from networks.CondConv import CondConv, DynamicConv
from .basic_blocks import *
from torchvision import models
from networks.attention_block import CBAMBlock,SEAttention
from networks.basic_blocks import Exchange,ModuleParallel,BatchNorm2dParallel
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
        self.conv192to64=nn.Conv2d(192, filters[0], kernel_size=1, stride=1, padding=0)
        self.bn_ycbr=nn.BatchNorm2d(filters[0])
        self.relu_ycbr=nn.ReLU(inplace=True)



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
        self.fem=FEM()
        self.fem_gps = FEM_gps()

        self.layer1_freq = self._make_layer(block, 64, blocks_num[0], bn_threshold)
        self.layer2_freq = self._make_layer(block, 128, blocks_num[1], bn_threshold, stride=2)
        self.layer3_freq = self._make_layer(block, 256, blocks_num[2], bn_threshold, stride=2)
        self.layer4_freq = self._make_layer(block, 512, blocks_num[3], bn_threshold, stride=2)
        self.dblock_freq = DBlock_parallel(filters[3], 2)
        self.decoder4_freq = DecoderBlock_parallel_exchange(filters[3], filters[2], 2, bn_threshold)
        self.decoder3_freq = DecoderBlock_parallel_exchange(filters[2], filters[1], 2, bn_threshold)
        self.decoder2_freq = DecoderBlock_parallel_exchange(filters[1], filters[0], 2, bn_threshold)
        self.decoder1_freq = DecoderBlock_parallel_exchange(filters[0], filters[0], 2, bn_threshold)
        self.finaldeconv1_freq = ModuleParallel(nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1))
        self.finalrelu1_freq = ModuleParallel(nn.ReLU(inplace=True))
        self.finalconv2_freq = ModuleParallel(nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1))
        self.finalrelu2_freq = ModuleParallel(nn.ReLU(inplace=True))
        self.se_freq = SEAttention(filters[0] // 2, reduction=4)
        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        # self.finalrelu1 = nonlinearity



        self.finaldeconv1 = ModuleParallel(nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1))
        self.finalrelu1 =  ModuleParallel(nn.ReLU(inplace=True))
        self.finalconv2 = ModuleParallel(nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1))
        self.finalrelu2 = ModuleParallel(nn.ReLU(inplace=True))
        self.se = SEAttention(filters[0] // 2, reduction=4)
        # self.se1 = SEAttention(filters[0] // 2, reduction=4)
        # self.atten=CBAMBlock(channel=filters[0], reduction=4, kernel_size=7)
        # self.fuse =NLBlockND_Fuse(filters[0]//2, filters[0]//2,mode='embedded', dimension=2)
        # self.fuse =CrissCrossAttention_Fuse(filters[0])
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

    def forward(self, inputs,ycbr):


        x = inputs[:, :3, :, :]
        g = inputs[:, 3:4, :, :]
        ycbr = DCT_Operation(ycbr)
        ycbr_dct=ycbr[:,64:,:,:]
        gps_dct = ycbr[:, :64, :, :]

        feat_DCT = self.conv192to64(self.fem(ycbr_dct))
        gps_dct =self.fem_gps(gps_dct)
        # print('feat_dctx:', feat_DCT.shape
        out_dct=feat_DCT,gps_dct
        # print('out2',gps_dct.shape)

        ##layers:
        x_1_f = self.layer1_freq(out_dct)
        x_2_f = self.layer2_freq(x_1_f)
        x_3_f = self.layer3_freq(x_2_f)
        x_4_f = self.layer4_freq(x_3_f)
        x_c_f = self.dblock_freq(x_4_f)
        # decoder
        x_d4_f = [self.decoder4_freq(x_c_f)[l] + x_3_f[l] for l in range(self.num_parallel)]
        x_d3_f = [self.decoder3_freq(x_d4_f)[l] + x_2_f[l] for l in range(self.num_parallel)]
        x_d2_f = [self.decoder2_freq(x_d3_f)[l] + x_1_f[l] for l in range(self.num_parallel)]
        x_d1_f = self.decoder1_freq(x_d2_f)
        x_out_f = self.finalrelu1_freq(self.finaldeconv1_freq(x_d1_f))
        x_out_f = self.finalrelu2_freq(self.finalconv2_freq(x_out_f))
        x_out_f[0] = self.se(x_out_f[0])
        x_out_f[1] = self.se(x_out_f[1])
        xoutf=torch.cat((x_out_f[0], x_out_f[1]), 1)


        # atten=self.atten(torch.cat((x_out[0], x_out[1]), 1))

        # out =self.finalconv(fuse)
        # out = self.finalconv(torch.cat((torch.cat((x_out[0], x_out[1]), 1),x_out[2]),1))
        out = self.finalconv(xoutf)
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
