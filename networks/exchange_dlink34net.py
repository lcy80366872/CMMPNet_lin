import torch.nn as nn
import torch
import math
from networks.CondConv import CondConv, DynamicConv
from .basic_blocks import *
from torchvision import models
from networks.attention_block import CBAMBlock, SEAttention


def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_parallel, bn_threshold, h, w, tile, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.height = conv2d_out_dim(h, kernel_size=3, stride=stride, padding=1)
        self.width = conv2d_out_dim(w, kernel_size=3, stride=stride, padding=1)
        self.mask_s1 = Mask_s(self.height, self.width, inplanes, tile, tile)  # 8浠ｈ〃8*8涓轰竴涓猤rid
        self.mask_s2 = Mask_s(self.height, self.width, inplanes, tile, tile)
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
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x
        # mask_s_1, norm_s, norm_s_t = self.mask_s1(x[0])
        # mask_s_2, norm_s, norm_s_t = self.mask_s2(x[1])

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        # mask_s1 = self.upsample(mask_s_1)
        # mask_s2 = self.upsample(mask_s_2)
        # print(mask_s1)

        out = self.conv2(out)
        out = self.bn2(out)

        # out[0] = out[0] * mask_s1 + out[1] * (1 - mask_s1)
        # out[1] = out[1] * mask_s2 + out[0] * (1 - mask_s2)
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
class decode_atten(nn.Module):
    def __init__(self, out_channels, kernel_size=3, patch_size=8,att_depth=3):
        super(decode_atten,self).__init__()
        self.patch_size = patch_size

        self.out_channels = out_channels
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                                stride=(self.patch_size, self.patch_size))
        self.att_depth = att_depth

    def forward(self, x, attentions):

        conv_feamap_size = nn.Conv2d(self.out_channels, self.out_channels,
                                     kernel_size=(2 ** self.att_depth, 2 ** self.att_depth),
                                     stride=(2 ** self.att_depth, 2 ** self.att_depth), groups=self.out_channels,
                                     bias=False)
        conv_feamap_size.weight = nn.Parameter(
            torch.ones((self.out_channels, 1, 2 ** self.att_depth, 2 ** self.att_depth)))
        conv_feamap_size.to(x.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False
        fold_layer = torch.nn.Fold(output_size=(x.size()[-2], x.size()[-1]),
                                   kernel_size=(self.patch_size, self.patch_size),
                                   stride=(self.patch_size, self.patch_size))
        argx_feamap = conv_feamap_size(x) / (2 ** self.att_depth * 2 ** self.att_depth)

        non_zeros = torch.unsqueeze(torch.count_nonzero(attentions, dim=-1) + 0.00001, dim=-1)

        att = torch.matmul(attentions/ non_zeros,
                           torch.unsqueeze(self.unfold(argx_feamap), dim=1).transpose(-1, -2))

        att = torch.squeeze(att, dim=1)

        att = fold_layer(att.transpose(-1, -2))

        return att

class atten(nn.Module):
    def __init__(self, out_channels, classes_num=1, patch_size=8, depth=5, att_depth=1, **kwargs):
        super(atten,self).__init__()
        self._depth = depth
        self._attention_on_depth=att_depth

        self._out_channels = out_channels

        self._in_channels = 3

        self.patch_size = patch_size

        self.conv_img=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7,7),padding=3),

            nn.Conv2d(64, classes_num, kernel_size=(3,3), padding=1)
        )

        self.conv_feamap=nn.Sequential(
            nn.Conv2d(self._out_channels, classes_num, kernel_size=(1, 1), stride=1)
        )

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

        self.resolution_trans=nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2*self.patch_size * self.patch_size, bias=False),
            nn.Linear(2*self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )

    def forward(self, x,fea):

        ini_img=self.conv_img(x)
        feamap = self.conv_feamap(fea) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)
        unfold_img = self.unfold(ini_img).transpose(-1, -2)
        unfold_img = self.resolution_trans(unfold_img)

        unfold_feamap = self.unfold(feamap)
        unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

        att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

        att=torch.unsqueeze(att,1)
        return att
class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_parallel=2,
                 num_classes=1,
                 bn_threshold=2e-2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.num_parallel = num_parallel

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

        self.layer1, h, w = self._make_layer(block, 64, blocks_num[0], bn_threshold, h, w, 8)
        self.layer2, h, w = self._make_layer(block, 128, blocks_num[1], bn_threshold, h, w, 4, stride=2)
        self.layer3, h, w = self._make_layer(block, 256, blocks_num[2], bn_threshold, h, w, 2, stride=2)
        self.layer4, h, w = self._make_layer(block, 512, blocks_num[3], bn_threshold, h, w, 1, stride=2)

        # self.dropout = ModuleParallel(nn.Dropout(p=0.5))

        self.dblock = DBlock_parallel(filters[3], 2)
        # self.dblock_add = DBlock(filters[3])
        # decoder
        self.decoder4 = DecoderBlock_parallel(filters[3], filters[2], 2)
        self.decoder3 = DecoderBlock_parallel(filters[2], filters[1], 2)
        self.decoder2 = DecoderBlock_parallel(filters[1], filters[0], 2)
        self.decoder1 = DecoderBlock_parallel(filters[0], filters[0], 2)

        # self.con1 = conv1x1(filters[0]*2,filters[0])
        # self.con2 = conv1x1(filters[1]*2, filters[1])
        # self.con3 = conv1x1(filters[2]*2, filters[2])

        # self.finaldeconv1_add = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        # self.finalrelu1_add = nonlinearity
        # self.finalconv2_add = nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1)
        # self.finalrelu2_add = nonlinearity

        self.finaldeconv1 = ModuleParallel(nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1))
        self.finalrelu1 = ModuleParallel(nn.ReLU(inplace=True))
        # self.finalrelu1 = nonlinearity
        self.finalconv2 = ModuleParallel(nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1))
        self.finalrelu2 = ModuleParallel(nn.ReLU(inplace=True))
        self.se = SEAttention(filters[0] // 2, reduction=4)
        # self.atten=CBAMBlock(channel=filters[0], reduction=4, kernel_size=7)
        self.finalconv = nn.Conv2d(filters[0], num_classes, 3, padding=1)
        self.afma = atten(out_channels=128,att_depth=3)
        self.decode_atten=decode_atten(out_channels=1)
        # -log((1-pi)/pi)
        # self.finalconv = ModuleParallel(nn.Conv2d(filters[0] // 2, num_classes, 3, padding=1))
        # self.alpha = nn.Parameter(torch.ones(num_parallel, requires_grad=True))
        # self.register_parameter('alpha', self.alpha)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # self.finalconv.bias.data.fill_(-2.19)

    def _make_layer(self, block, planes, num_blocks, bn_threshold, h, w, tile, stride=1):
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
            layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, h, w, tile))

        return nn.Sequential(*layers), h, w

    def forward(self, inputs):

        x = inputs[:, :3, :, :]
        g = inputs[:, 3:, :, :]
        orign_x= x


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
        atten = self.afma(orign_x,x_2[0])
        # print(atten.shape)
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

        x_out[0] = self.se(x_out[0])
        x_out[1] = self.se(x_out[1])
        # atten=self.atten(torch.cat((x_out[0], x_out[1]), 1))
        out = self.finalconv(torch.cat((x_out[0], x_out[1]), 1))

        out = self.decode_atten(out,atten)*out+out

        # out=self.finalconv(x_out)
        # alpha_soft = F.softmax(self.alpha,dim=0)
        # ens = 0
        # for l in range(self.num_parallel):
        #     ens += alpha_soft[l] * out[l].detach()
        out = torch.sigmoid(out)
        # print(out)
        # out =nn.LogSoftmax()(ens)
        # out.append(ens)#[忙露鈥溍ｂ€氣€灻┾€∨撁β澦喢︹€櫬趁ヂ忊€犆┞惵ニ喡紆t忙碌 茫茠楼氓录路忙碌 忙 娄忙禄鈥樏┡铰该ヂ÷pha茅聧搂氓鈥郝ｂ偓鈧┞嵟∶ヂ郝Ｃβ€榦utput,忙露鈥溍⑩€毬┞嵚徝ヂ徦溍伱β垛€溍捖�

        return out,atten


def DinkNet34_CMMPNet():
    model = ResNet(block=BasicBlock, blocks_num=[3, 4, 6, 3], num_parallel=2, num_classes=1, bn_threshold=2e-2)
    return model
