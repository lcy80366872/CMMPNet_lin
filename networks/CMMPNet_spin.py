import torch
import torch.nn as nn
from torchvision import models
from .basic_blocks import *
import math
import torch.nn.functional as F
from networks.MPM import SPHead


up_kwargs = {'mode': 'bilinear', 'align_corners': True}

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d



class SpatialGCN(nn.Module):
    """
        Spatial Space Graph Reasoning ...
    """
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)

        out = F.relu_(self.out(AVW) + x)

        return out



class spin(nn.Module):
    """
        Spatial and Interaction Space Graph Reasoning ...
    """
    def __init__(self, planes, ratio=4):
        super(spin, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes))
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        # Sum features ...
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out

# spp只到拼接，后面的fc层步骤不算在spp内
class SPPLayer(torch.nn.Module):
    def __init__(self, block_size=[1, 2, 4], pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.block_size = block_size
        self.pool_type = pool_type
        self.spp = self.make_spp(out_pool_size=self.block_size, pool_type=self.pool_type)

    def make_spp(self, out_pool_size, pool_type='maxpool'):
        func = []
        for i in range(len(out_pool_size)):
            if pool_type == 'max_pool':
                func.append(nn.AdaptiveMaxPool2d(output_size=(out_pool_size[i], out_pool_size[i])))
            if pool_type == 'avg_pool':
                func.append(nn.AdaptiveAvgPool2d(output_size=(out_pool_size[i], out_pool_size[i])))
        return func

    def forward(self, x):
        num = x.size(0)
        for i in range(len(self.block_size)):
            # view：返回一个有相同数据但大小不同的tensor。 返回的tensor必须有与原tensor相同的数据和相同数目的元素，但可以有不同的大小，大小可以自己选
            # 将spp每个分支展成一维向量，再拼接
            tensor = self.spp[i](x).view(num, -1)
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                # 在第1维度拼接，也就是横向拼接成长条
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)

        return x_flatten


class DEM(torch.nn.Module):  # Dual Enhancement Module
    def __init__(self, channel, block_size=[1, 2, 4]):
        super(DEM, self).__init__()
        # 这里相当于1*1卷积了 ，padding=0
        self.rgb_local_message = self.local_message_prepare(channel, 1, 1, 0)  # 文中的L
        self.add_local_message = self.local_message_prepare(channel, 1, 1, 0)

        self.rgb_spp = SPPLayer(block_size=block_size)
        self.add_spp = SPPLayer(block_size=block_size)
        self.rgb_global_message = self.global_message_prepare(block_size, channel)  # 文中的G
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
        rgb_global_info = torch.unsqueeze(torch.unsqueeze(self.rgb_global_message(self.rgb_spp(rgb_local_info)), -1),
                                          -1).expand(rgb_local_info.size())
        add_global_info = torch.unsqueeze(torch.unsqueeze(self.add_global_message(self.add_spp(add_local_info)), -1),
                                          -1).expand(add_local_info.size())
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

        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1 = nonlinearity

        # Dual Graph Convolutional module
        self.dgcn_seg1 = spin(filters[0] // 2, ratio=2)
        self.dgcn_seg2 = spin(filters[0] // 2, ratio=2)
        self.dgcn_seg3 = spin(filters[0] // 2, ratio=2)

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
        x = self.firstconv1(x)
        add = self.firstconv1_add(add)
        x = self.firstmaxpool(self.firstrelu(self.firstbn(x)))
        add = self.firstmaxpool_add(self.firstrelu_add(self.firstbn_add(add)))
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
        # x_e4 = self.head(x_e4)
        # add_e4 = self.head(add_e4)
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

        # SPIN Pyramid
        spin257 = self.dgcn_seg1(x_out)  # SPIN at 257x257 scale
        f1_128 = self.maxpool(x_out)

        spin128 = self.dgcn_seg2(f1_128)  # SPIN at 128*128 scale
        f1_64 = self.maxpool(f1_128)
        spin64 = self.dgcn_seg3(f1_64)  # SPIN at 64*64 scale
        spin64_up = F.interpolate(spin64, size=(f1_128.shape[2], f1_128.shape[3]), mode="bilinear")
        spin128_comb = torch.add(spin64_up, spin128)
        spin128_up = F.interpolate(spin128_comb, size=(spin257.shape[2], spin257.shape[3]), mode="bilinear")
        f1 = torch.add(spin128_up, spin257)

        # SPIN Pyramid_add
        spin257 = self.dgcn_seg1(add_out)  # SPIN at 257x257 scale
        f2_128 = self.maxpool(add_out)
        spin128 = self.dgcn_seg2(f2_128)  # SPIN at 128*128 scale
        f2_64 = self.maxpool(f2_128)
        spin64 = self.dgcn_seg3(f2_64)  # SPIN at 64*64 scale
        spin64_up = F.interpolate(spin64, size=(f2_128.shape[2], f2_128.shape[3]), mode="bilinear")
        spin128_comb = torch.add(spin64_up, spin128)
        spin128_up = F.interpolate(spin128_comb, size=(spin257.shape[2], spin257.shape[3]), mode="bilinear")
        f2 = torch.add(spin128_up, spin257)

        x_out = self.finalrelu2(self.finalconv2(f1))
        add_out = self.finalrelu2_add(self.finalconv2_add(f2))

        out = self.finalconv(torch.cat((x_out, add_out), 1))  # b*1*h*w
        return torch.sigmoid(out)




