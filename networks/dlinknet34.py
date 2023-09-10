import torch
from torchvision import models
from .basic_blocks import *
import math
import torch.nn.functional as F
from networks.MPM import SPHead


class DinkNet34_CMMPNet(nn.Module):
    def __init__(self, block_size='1,2,4'):
        super(DinkNet34_CMMPNet, self).__init__()
        filters = [64, 128, 256, 512]
        self.net_name = "CMMPnet"
        self.block_size = [int(s) for s in block_size.split(',')]

        # img
        resnet = models.resnet34(pretrained=True)

        self.firstconv1 = nn.Conv2d(1, filters[0], kernel_size=7, stride=2, padding=3)

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.firstconv1_add = nn.Conv2d(1, filters[0], kernel_size=7, stride=2, padding=3)

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DBlock(filters[3])

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], filters[0], 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.finalrelu2 = nonlinearity

        self.finalconv = nn.Conv2d(filters[0], 1, 3, padding=1)

    def forward(self, inputs):
        x = inputs[:, :3, :, :]  # image
        add = inputs[:, 3:, :, :]  # gps_map or lidar_map
        # 杩涘叆缂栫爜-瑙ｇ爜缁撴瀯鍓嶆湁涓皢鍘熷浘鍍忓仛鍗风Н姝ラ
        x = self.firstconv1(add)
        x = self.firstmaxpool(self.firstrelu(self.firstbn(x)))
        # 姣忎竴灞傜殑鍥惧儚鍜宎dding鐨勯澶栦俊鎭緥濡俫ps閮借緭鍏EM妯″潡锛岃緭鍑哄寮虹殑鍥惧儚鍜宎dding鐗瑰緛淇℃伅锛岀劧鍚庡啀杈撳叆涓嬩竴灞備互姝ゅ惊鐜�
        x_e1 = self.encoder1(x)
        x_e2 = self.encoder2(x_e1)
        x_e3 = self.encoder3(x_e2)
        x_e4 = self.encoder4(x_e3)

        x_c = self.dblock(x_e4)

        # 浼犻€掑寮轰俊鎭椂杩樻湁璺宠穬杩炴帴
        # Decoder
        x_d4 = self.decoder4(x_c) + x_e3
        x_d3 = self.decoder3(x_d4) + x_e2
        x_d2 = self.decoder2(x_d3) + x_e1
        x_d1 = self.decoder1(x_d2)

        x_out = self.finalrelu1(self.finaldeconv1(x_d1))
        x_out = self.finalrelu2(self.finalconv2(x_out))

        out = self.finalconv(x_out)  # b*1*h*w

        return torch.sigmoid(out)
