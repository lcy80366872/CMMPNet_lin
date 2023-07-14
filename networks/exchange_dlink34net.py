import torch.nn as nn
import torch
import  math
from networks.CondConv import CondConv, DynamicConv
from .basic_blocks import *
from torchvision import models
from networks.attention_block import CBAMBlock,SEAttention

def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, h, w,tile,stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.height = conv2d_out_dim(h, kernel_size=3, stride=stride, padding=1)
        self.width = conv2d_out_dim(w, kernel_size=3, stride=stride, padding=1)
        self.mask_s1 = Mask_s(self.height, self.width, planes,tile, tile)#8代表8*8为一个grid
        self.mask_s2 = Mask_s(self.height, self.width, planes, tile, tile)
        self.upsample = nn.Upsample(size=(self.height, self.width), mode='nearest')

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
        # self.bn1_list = []
        # for module in self.bn1.modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         self.bn1_list.append(module)
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x
        mask_s_1, norm_s, norm_s_t = self.mask_s1(x[0])
        mask_s_2, norm_s, norm_s_t = self.mask_s2(x[1])

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)
        
        # if len(x) > 1:
        #     out = self.exchange(out, self.bn1_list, self.bn_threshold)
        # mask_s_1, norm_s, norm_s_t = self.mask_s1(out[0])
        # mask_s_2, norm_s, norm_s_t = self.mask_s2(out[1])
        mask_s1 = self.upsample(mask_s_1)
        mask_s2 = self.upsample(mask_s_2)
       
        out = self.conv2(out)
        out = self.bn2(out)
        out[0] = out[0]  * mask_s1+out[1]*(1-mask_s1)
        out[1] = out[1] * mask_s2 +out[0]*(1-mask_s2)
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
        h = conv2d_out_dim(512, kernel_size=7, stride=2, padding=3)
        w = conv2d_out_dim(512, kernel_size=7, stride=2, padding=3)
        h = conv2d_out_dim(h, kernel_size=3, stride=2, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, stride=2, padding=1)

        self.layer1,h,w = self._make_layer(block, 64, blocks_num[0], bn_threshold,h,w,8)
        self.layer2,h,w = self._make_layer(block, 128, blocks_num[1], bn_threshold,h,w,4, stride=2)
        self.layer3,h,w = self._make_layer(block, 256, blocks_num[2], bn_threshold,h,w,2, stride=2)
        self.layer4,h,w = self._make_layer(block, 512, blocks_num[3], bn_threshold,h,w,1, stride=2)

        # self.dropout = ModuleParallel(nn.Dropout(p=0.5))

        self.dblock = DBlock_parallel(filters[3],2)
        # self.dblock_add = DBlock(filters[3])
        # decoder
        self.decoder4 = DecoderBlock_parallel(filters[3], filters[2],2)
        self.decoder3 = DecoderBlock_parallel(filters[2], filters[1],2)
        self.decoder2 = DecoderBlock_parallel(filters[1], filters[0],2)
        self.decoder1 = DecoderBlock_parallel(filters[0], filters[0],2)

        # self.con1 = conv1x1(filters[0]*2,filters[0])
        # self.con2 = conv1x1(filters[1]*2, filters[1])
        # self.con3 = conv1x1(filters[2]*2, filters[2])


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
        self.finalconv = nn.Conv2d(filters[0], num_classes, 3, padding=1)

         #-log((1-pi)/pi)
        # self.finalconv = ModuleParallel(nn.Conv2d(filters[0] // 2, num_classes, 3, padding=1))
        # self.alpha = nn.Parameter(torch.ones(num_parallel, requires_grad=True))
        # self.register_parameter('alpha', self.alpha)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # self.finalconv.bias.data.fill_(-2.19)
    def _make_layer(self, block, planes, num_blocks, bn_threshold,h,w,tile, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                BatchNorm2dParallel(planes * block.expansion, self.num_parallel)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, h, w, tile, stride, downsample))
        h = conv2d_out_dim(h, kernel_size=1, stride=stride, padding=0)
        w = conv2d_out_dim(w, kernel_size=1, stride=stride, padding=0)
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold,h,w,tile))

        return nn.Sequential(*layers),h,w

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

        # x_d4= self.con3([torch.cat((self.decoder4(x_c)[l], x_3[l]), 1) for l in range(self.num_parallel)])
        # x_d3 = self.con2([torch.cat((self.decoder3(x_d4)[l], x_2[l]), 1) for l in range(self.num_parallel)])
        # x_d2 = self.con1([torch.cat((self.decoder2(x_d3)[l], x_1[l]), 1) for l in range(self.num_parallel)])
        # x_d1 = self.decoder1(x_d2)

        x_out = self.finalrelu1(self.finaldeconv1(x_d1))
        x_out = self.finalrelu2(self.finalconv2(x_out))

        x_out[0]=self.se(x_out[0])
        x_out[1] = self.se(x_out[1])
        # atten=self.atten(torch.cat((x_out[0], x_out[1]), 1))
        out = self.finalconv(torch.cat((x_out[0], x_out[1]), 1))
        # out=self.finalconv(x_out)
        # alpha_soft = F.softmax(self.alpha,dim=0)
        # ens = 0
        # for l in range(self.num_parallel):
        #     ens += alpha_soft[l] * out[l].detach()
        out = torch.sigmoid(out)
        # print(out)
        # out =nn.LogSoftmax()(ens)
        # out.append(ens)#[æ¶“ã‚„é‡œæˆæ’³å†é¨åˆ¼utæµ ãƒ¥å¼·æµ æ ¦æ»‘éŽ¸å¡§lphaé§å›ªã€€éšåº£æ®‘output,æ¶“â‚¬éå˜ç¬æ¶“çŒ

        return out


def DinkNet34_CMMPNet():
    model = ResNet(block=BasicBlock, blocks_num=[3, 4, 6, 3],num_parallel=2,num_classes=1,bn_threshold=2e-2)
    return model
