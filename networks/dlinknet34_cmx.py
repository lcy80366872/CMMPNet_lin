import torch
from torchvision import models
from networks.basic_blocks import *
import math
import torch.nn.functional as F
from networks.net_utils import *

class DinkNet34_CMMPNet(nn.Module):
    def __init__(self, block_size='1,2,4'):
        super(DinkNet34_CMMPNet, self).__init__()
        filters = [64, 128, 256, 512]
        num_heads = [1, 2, 4, 8]
        self.net_name = "CMMPnet"
        self.block_size = [int(s) for s in block_size.split(',')]

        # img
        resnet = models.resnet34(pretrained=False)
        resnet1 = models.resnet34(pretrained=False)
        self.firstconv1 = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.firstconv1_add = nn.Conv2d(1, filters[0], kernel_size=7, stride=2, padding=3)
        self.firstbn_add = resnet1.bn1
        self.firstrelu_add = resnet1.relu
        self.firstmaxpool_add = resnet1.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.frm1=FeatureRectifyModule(filters[0])
        self.frm2 = FeatureRectifyModule(filters[1])
        self.frm3 = FeatureRectifyModule(filters[2])
        self.frm4 = FeatureRectifyModule(filters[3])
        self.encoder1_add = resnet1.layer1
        self.encoder2_add = resnet1.layer2
        self.encoder3_add = resnet1.layer3
        self.encoder4_add = resnet1.layer4
        self.ffm1 = FeatureFusionModule(filters[0], num_heads=num_heads[0])
        self.ffm2 = FeatureFusionModule(filters[1], num_heads=num_heads[1])
        self.ffm3 = FeatureFusionModule(filters[2], num_heads=num_heads[2])
        self.ffm4=FeatureFusionModule(filters[3],num_heads=num_heads[3])
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1)
        self.finalrelu2 = nonlinearity



        self.finaldeconv1_add = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1_add = nonlinearity
        self.finalconv2_add = nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1)
        self.finalrelu2_add = nonlinearity

        self.finalconv = nn.Conv2d(filters[0]//2, 1, 3, padding=1)

    def forward(self, inputs):
        x = inputs[:, :3, :, :]  # image
        add = inputs[:, 3:, :, :]  # gps_map or lidar_map
        # 进入编码-解码结构前有个将原图像做卷积步骤
        x = self.firstconv1(x)
        x = self.firstmaxpool(self.firstrelu(self.firstbn(x)))
        add = self.firstconv1_add(add)
        add = self.firstmaxpool_add(self.firstrelu_add(self.firstbn_add(add)))
        # 每一层的图像和adding的额外信息例如gps都输入DEM模块，输出增强的图像和adding特征信息，然后再输入下一层以此循环
        x_e1 = self.encoder1(x)
        add_e1 = self.encoder1_add(add)
        x_e1,add_e1=self.frm1(x_e1,add_e1)
        x_fused1 = self.ffm1(x_e1,add_e1)

        x_e2 = self.encoder2(x_e1)
        add_e2 = self.encoder2_add(add_e1)
        x_e2, add_e2 = self.frm2(x_e2, add_e2)
        x_fused2 = self.ffm2(x_e2, add_e2)

        x_e3 = self.encoder3(x_e2)
        add_e3 = self.encoder3_add(add_e2)
        x_e3, add_e3 = self.frm3(x_e3, add_e3)
        x_fused3 = self.ffm3(x_e3, add_e3)

        x_e4 = self.encoder4(x_e3)
        add_e4 = self.encoder4_add(add_e3)
        x_c, add_c = self.frm4(x_e4, add_e4)
        x_fused4 = self.ffm1(x_e4, add_e4)


        # 传递增强信息时还有跳跃连接
        # Decoder
        x_d4 = self.decoder4(x_fused4) + x_fused3
        x_d3 = self.decoder3(x_d4) + x_fused2
        x_d2 = self.decoder2(x_d3) + x_fused1
        x_d1 = self.decoder1(x_d2)



        x_out = self.finalrelu1(self.finaldeconv1(x_d1))
        x_out   = self.finalrelu2(self.finalconv2(x_out))


        out = self.finalconv(x_out)  # b*1*h*w

        return torch.sigmoid(out)

