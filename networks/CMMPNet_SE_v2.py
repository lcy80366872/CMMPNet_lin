import torch
from torchvision import models
from .basic_blocks import *
import math
import torch.nn.functional as F
# from networks.MPM import SPHead
from networks.connect import build_connect
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
        self.filter = FilterLayer(2*in_planes, out_planes, reduction)

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

        self.gate_rgb = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)
        self.gate_hha = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x,y):
        rgb, hha = x,y
        b, c, h, w = rgb.size()

        rec_rgb = self.fsp_rgb(hha, rgb)
        rec_hha = self.fsp_hha(rgb, hha)

        cat_fea = torch.cat([rec_rgb, rec_hha], dim=1)

        attention_vector_l = self.gate_rgb(cat_fea)
        attention_vector_r = self.gate_hha(cat_fea)

        attention_vector = torch.cat([attention_vector_l, attention_vector_r], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        merge_feature = rgb*attention_vector_l + hha*attention_vector_r

        rgb_out = (rgb + merge_feature) / 2
        hha_out = (hha + merge_feature) / 2

        rgb_out = self.relu1(rgb_out)
        hha_out = self.relu2(hha_out)
        merge = self.relu1(merge_feature)

        return merge,rgb_out, hha_out
def fuse_operation(x,add,channel):
    a=torch.cat((x, add), 1)
    f=nn.Conv2d(channel*2,channel,3,1,1)
    return f



class DinkNet34_CMMPNet(nn.Module):
    def __init__(self, block_size='1,2,4'):
        super(DinkNet34_CMMPNet, self).__init__()
        filters = [64, 128, 256, 512]
        self.net_name = "CMMPnet"
        self.block_size = [int(s) for s in block_size.split(',')]
        self.connect = build_connect(1, 9, nn.BatchNorm2d)  # 1和9分别代表类别，和生成连接图的层数（mask8个方向生成的连接图+mask自己本身）

        # img
        resnet = models.resnet34(pretrained=True)
        self.firstconv1 = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

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

        ## addinfo, e.g, gps_map, lidar_map
        resnet1 = models.resnet34(pretrained=True)
        self.firstconv1_add = nn.Conv2d(1, filters[0], kernel_size=7, stride=2, padding=3)
        self.firstbn_add = resnet1.bn1
        self.firstrelu_add = resnet1.relu
        self.firstmaxpool_add = resnet1.maxpool

        self.encoder1_add = resnet1.layer1
        self.encoder2_add = resnet1.layer2
        self.encoder3_add = resnet1.layer3
        self.encoder4_add = resnet1.layer4

        self.dblock_add = DBlock(filters[3])
        # self.head = SPHead(filters[3], filters[3], nn.BatchNorm2d, up_kwargs)

        self.decoder4_add = DecoderBlock1DConv4(filters[3], filters[2])
        self.decoder3_add = DecoderBlock1DConv4(filters[2], filters[1])
        self.decoder2_add = DecoderBlock1DConv4(filters[1], filters[0])
        self.decoder1_add = DecoderBlock1DConv4(filters[0], filters[0])

        self.finaldeconv1_add = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1_add = nonlinearity
        self.finalconv2_add = nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1)
        self.finalrelu2_add = nonlinearity

        ### DEM
        self.dem_e1 = DEM(filters[0], filters[0])
        self.dem_e2 = DEM(filters[1], filters[1])
        self.dem_e3 = DEM(filters[2], filters[2])
        self.dem_e4 = DEM(filters[3], filters[3])

        self.dem_d4 = DEM(filters[2], filters[2])
        self.dem_d3 = DEM(filters[1], filters[1])
        self.dem_d2 = DEM(filters[0], filters[0])
        self.dem_d1 = DEM(filters[0], filters[0])

        self.finalconv = nn.Conv2d(filters[0]//2, 1, 3, padding=1)

    def forward(self, inputs):
        x = inputs[:, :3, :, :]  # image
        add = inputs[:, 3:, :, :]  # gps_map or lidar_map
        # 进入编码-解码结构前有个将原图像做卷积步骤
        x = self.firstconv1(x)
        add = self.firstconv1_add(add)
        x = self.firstmaxpool(self.firstrelu(self.firstbn(x)))
        add = self.firstmaxpool_add(self.firstrelu_add(self.firstbn_add(add)))
        # 每一层的图像和adding的额外信息例如gps都输入DEM模块，输出增强的图像和adding特征信息，然后再输入下一层以此循环
        x_e1 = self.encoder1(x)
        add_e1 = self.encoder1_add(add)
        merge1,x_e1, add_e1 = self.dem_e1(x_e1, add_e1)


        x_e2 = self.encoder2(x_e1)
        add_e2 = self.encoder2_add(add_e1)
        merge2,x_e2, add_e2 = self.dem_e2(x_e2, add_e2)


        x_e3 = self.encoder3(x_e2)
        add_e3 = self.encoder3_add(add_e2)
        merge3,x_e3, add_e3 = self.dem_e3(x_e3, add_e3)

        x_e4 = self.encoder4(x_e3)
        add_e4 = self.encoder4_add(add_e3)
        merge4,x_e4, add_e4 = self.dem_e4(x_e4, add_e4)

        # Center
        # x_e4  = self.head(x_e4)
        # add_e4= self.head(add_e4)
        x_add = self.dblock(merge4)

        # 传递增强信息时还有跳跃连接
        # Decoder
        x_add_d4 = self.decoder4(x_add)
        x_add_d3 = self.decoder3(x_add_d4) + merge2
        x_add_d2 = self.decoder2(x_add_d3) + merge1
        x_add_d1 = self.decoder1(x_add_d2)
        x_out = self.finalrelu1(self.finaldeconv1(x_add_d1))
        x_out = self.finalrelu2(self.finalconv2(x_out))
        # x_out, connect, connect_d1 = self.connect(x_out)

        out = self.finalconv(x_out)  # b*1*h*w
        return torch.sigmoid(out)#,torch.sigmoid(connect), torch.sigmoid(connect_d1)

