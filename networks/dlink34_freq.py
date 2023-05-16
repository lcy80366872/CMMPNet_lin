import torch
from torchvision import models
from .basic_blocks import *
import math
import torch.nn.functional as F
from networks.MPM import SPHead
from networks.Freq import *
up_kwargs = {'mode': 'bilinear', 'align_corners': True}



class DinkNet34_CMMPNet(nn.Module):
    def __init__(self, block_size='1,2,4'):
        super(DinkNet34_CMMPNet, self).__init__()
        filters = [64, 128, 256, 512]
        self.net_name = "CMMPnet"
        self.block_size = [int(s) for s in block_size.split(',')]

        # img
        resnet = models.resnet34(pretrained=True)
        self.conv192to64 = nn.Conv2d(192, filters[0], kernel_size=1, stride=1, padding=0)
        self.bn_ycbr = nn.BatchNorm2d(filters[0])
        self.relu_ycbr = nn.ReLU(inplace=True)
        self.fem = FEM()

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DBlock(filters[3])
        # self.head = SPHead(filters[3], filters[3], nn.BatchNorm2d, up_kwargs)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1)
        self.finalrelu2 = nonlinearity


        self.finalconv = nn.Conv2d(filters[0]//2, 1, 3, padding=1)

    def forward(self, inputs,ycbr):
        x = inputs[:, :3, :, :]  # image
        add = inputs[:, 3:, :, :]  # gps_map or lidar_map
        # 进入编码-解码结构前有个将原图像做卷积步骤
        ycbr = DCT_Operation(ycbr)
        ycbr_dct = ycbr[:, 64:, :, :]
        feat_DCT = self.conv192to64(self.fem(ycbr_dct))
        feat_DCT =self.relu_ycbr(self.bn_ycbr(feat_DCT))


        # 每一层的图像和adding的额外信息例如gps都输入DEM模块，输出增强的图像和adding特征信息，然后再输入下一层以此循环
        x_e1 = self.encoder1(feat_DCT)
        x_e2 = self.encoder2(x_e1)
        x_e3 = self.encoder3(x_e2)
        x_e4 = self.encoder4(x_e3)

        x_c = self.dblock(x_e4)
        # 传递增强信息时还有跳跃连接
        # Decoder
        x_d4 = self.decoder4(x_c) + x_e3
        x_d3 = self.decoder3(x_d4) + x_e2
        x_d2 = self.decoder2(x_d3) + x_e1
        x_d1 = self.decoder1(x_d2)

        x_out = self.finalrelu1(self.finaldeconv1(x_d1))
        x_out = self.finalrelu2(self.finalconv2(x_out))
        out = self.finalconv(x_out)
        return torch.sigmoid(out)

