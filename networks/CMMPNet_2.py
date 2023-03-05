import torch
from torchvision import models
from .basic_blocks import *
import math
import torch.nn.functional as F
from networks.MPM import SPHead
from networks.MPM import StripPooling
up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class DEM(torch.nn.Module):  # Dual Enhancement Module
    def __init__(self, channel, block_size=[1, 2, 4]):
        super(DEM, self).__init__()
        # 这里相当于1*1卷积了 ，padding=0
        self.rgb_local_message = self.local_message_prepare(channel, 1, 1, 0)
        self.add_local_message = self.local_message_prepare(channel, 1, 1, 0)

#         self.rgb_spp = StripPooling(channel, (4, 2), nn.BatchNorm2d, up_kwargs)
#         self.add_spp = StripPooling(channel, (4, 2), nn.BatchNorm2d, up_kwargs)
#         self.rgb_global_message = self.rgb_spp
#         self.add_global_message = self.add_spp
        self.rgb_spp = SPPLayer(block_size=block_size)
        self.add_spp = SPPLayer(block_size=block_size)
        self.rgb_global_message = self.global_message_prepare(block_size, channel) #文中的G
        self.add_global_message = self.global_message_prepare(block_size, channel)

        self.rgb_local_gate = self.gate_build(channel * 2, channel, 1, 1, 0)
        self.rgb_global_gate = self.gate_build(channel * 2, channel, 1, 1, 0)

        self.add_local_gate = self.gate_build(channel * 2, channel, 1, 1, 0)
        self.add_global_gate = self.gate_build(channel * 2, channel, 1, 1, 0)

    def local_message_prepare(self, dim, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(dim)
        )

    # 下面做的是spp得到拼接向量之后的步骤FC+relu
    def global_message_prepare(self, block_size, dim):
        num_block = 0
        for i in block_size:
            num_block += i * i
        return nn.Sequential(
            nn.Linear(num_block * dim, dim),
            nn.ReLU()
        )

    def gate_build(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, rgb_info, add_info):
        rgb_local_info = self.rgb_local_message(rgb_info)
        add_local_info = self.add_local_message(add_info)
        # 就是在指定的位置插入一个维度，有两个参数，input是输入的tensor,dim是要插到的维度
        # https://blog.csdn.net/ljwwjl/article/details/115342632
        # SPP+FC+RELU+扩展成（N*c*1*1）+复制成N*c*h*w
        rgb_global_info = self.rgb_global_message(rgb_local_info)
        add_global_info = self.add_global_message(add_local_info)
        # add_local_gate的输出大小也为N*C*H*W
        rgb_info = rgb_info + add_local_info * self.add_local_gate(
            torch.cat((add_local_info, add_global_info), 1)) + add_global_info * self.add_global_gate(
            torch.cat((add_local_info, add_global_info), 1))
        add_info = add_info + rgb_local_info * self.rgb_local_gate(
            torch.cat((rgb_local_info, rgb_global_info), 1)) + rgb_global_info * self.rgb_global_gate(
            torch.cat((rgb_local_info, rgb_global_info), 1))

        return rgb_info, add_info


class DinkNet34_CMMPNet(nn.Module):
    def __init__(self, block_size='1,2,4'):
        super(DinkNet34_CMMPNet, self).__init__()
        filters = [64, 128, 256, 512]
        self.net_name = "CMMPnet"
        self.block_size = [int(s) for s in block_size.split(',')]
        self.firstlayer_img = StripConvBlock(3, filters[0], nn.BatchNorm2d)
        self.firstlayer_add = StripConvBlock(1, filters[0], nn.BatchNorm2d)
        # img
        resnet = models.resnet34(pretrained=True)
#         self.firstconv1 = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DBlock(filters[3])
        self.head = SPHead(filters[3], filters[3], nn.BatchNorm2d, up_kwargs)

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
        self.head = SPHead(filters[3], filters[3], nn.BatchNorm2d, up_kwargs)

        self.decoder4_add = DecoderBlock(filters[3], filters[2])
        self.decoder3_add = DecoderBlock(filters[2], filters[1])
        self.decoder2_add = DecoderBlock(filters[1], filters[0])
        self.decoder1_add = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1_add = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1_add = nonlinearity
        self.finalconv2_add = nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1)
        self.finalrelu2_add = nonlinearity

        ### DEM
        self.dem_e1 = DEM(filters[0], self.block_size)
        self.dem_e2 = DEM(filters[1], self.block_size)
        self.dem_e3 = DEM(filters[2], self.block_size)
        self.dem_e4 = DEM(filters[3], self.block_size)

        self.dem_d4 = DEM(filters[2], self.block_size)
        self.dem_d3 = DEM(filters[1], self.block_size)
        self.dem_d2 = DEM(filters[0], self.block_size)
        self.dem_d1 = DEM(filters[0], self.block_size)

        self.finalconv = nn.Conv2d(filters[0], 1, 3, padding=1)

    def forward(self, inputs):
        x = inputs[:, :3, :, :]  # image
        add = inputs[:, 3:, :, :]  # gps_map or lidar_map
        # 进入编码-解码结构前有个将原图像做卷积步骤
        x = self.firstlayer_img(x)
        add = self.firstlayer_add(add)
        # 每一层的图像和adding的额外信息例如gps都输入DEM模块，输出增强的图像和adding特征信息，然后再输入下一层以此循环
        x_e1 = self.encoder1(x)
        add_e1 = self.encoder1_add(add)
        x_e1, add_e1 = self.dem_e1(x_e1, add_e1)

        x_e2 = self.encoder2(x_e1)
        add_e2 = self.encoder2_add(add_e1)
        x_e2, add_e2 = self.dem_e2(x_e2, add_e2)

        x_e3 = self.encoder3(x_e2)
        add_e3 = self.encoder3_add(add_e2)
        x_e3, add_e3 = self.dem_e3(x_e3, add_e3)

        x_e4 = self.encoder4(x_e3)
        add_e4 = self.encoder4_add(add_e3)
        x_e4, add_e4 = self.dem_e4(x_e4, add_e4)

        # Center

        x_c = self.dblock(x_e4)
        add_c = self.dblock_add(add_e4)
        # 传递增强信息时还有跳跃连接
        # Decoder
        x_d4 = self.decoder4(x_c) + x_e3
        add_d4 = self.decoder4_add(add_c) + add_e3
        x_d4, add_d4 = self.dem_d4(x_d4, add_d4)

        x_d3 = self.decoder3(x_d4) + x_e2
        add_d3 = self.decoder3_add(add_d4) + add_e2
        x_d3, add_d3 = self.dem_d3(x_d3, add_d3)

        x_d2 = self.decoder2(x_d3) + x_e1
        add_d2 = self.decoder2_add(add_d3) + add_e1
        x_d2, add_d2 = self.dem_d2(x_d2, add_d2)

        x_d1 = self.decoder1(x_d2)
        add_d1 = self.decoder1_add(add_d2)
        x_d1, add_d1 = self.dem_d1(x_d1, add_d1)

        x_out = self.finalrelu1(self.finaldeconv1(x_d1))
        add_out = self.finalrelu1_add(self.finaldeconv1_add(add_d1))
        x_out = self.finalrelu2(self.finalconv2(x_out))
        add_out = self.finalrelu2_add(self.finalconv2_add(add_out))

        out = self.finalconv(torch.cat((x_out, add_out), 1))  # b*1*h*w
        return torch.sigmoid(out)




class StripConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm, inp=False):
        super(StripConvBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        # self.bn1 = BatchNorm(in_channels // 4)
        # self.relu1 = nn.ReLU()
        self.inp = inp

        self.stripconv1 = nn.Conv2d(
            in_channels , out_channels // 8, (1, 9), padding=(0, 4)
        )
        self.stripconv2  = nn.Conv2d(
            in_channels , out_channels // 8, (9, 1), padding=(4, 0)
        )
        self.stripconv3= nn.Conv2d(
            in_channels , out_channels // 8, (9, 1), padding=(4, 0)
        )
        self.stripconv4 = nn.Conv2d(
            in_channels , out_channels // 8, (1, 9), padding=(0, 4)
        )

        self.bn2 = BatchNorm(out_channels //2 )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels//2, out_channels, 1)
        self.bn3 = BatchNorm(out_channels)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp = False):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu1(x)

        x1 = self.stripconv1(x)
        x2 = self.stripconv2(x)
        x3 = self.inv_h_transform(self.stripconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.stripconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    #是输出相应的列,[...,1]表示数据的第二列；[...,0]表示数据的第二列,a[:-1]表示从第0位开始直到最后一位的前一位
    def h_transform(self, x):
        shape = x.size()   #N*C*H*W
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x
    #应该是与h_trans相反的过程，再转换回去
    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)
