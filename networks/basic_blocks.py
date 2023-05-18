import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)
class Exchange_3(nn.Module):
    def __init__(self):
        super(Exchange_3, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2,bn3 = bn[0].weight.abs(), bn[1].weight.abs(),bn[2].weight.abs()
        #就是大于阈值的那些通道保留，小于阈值的那些通道取另外一个的值
        x1, x2 ,x3= torch.zeros_like(x[0]), torch.zeros_like(x[1]),torch.zeros_like(x[2])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = (x[1][:, bn1 < bn_threshold]+x[2][:, bn1 < bn_threshold])/2.0
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = (x[0][:, bn2 < bn_threshold]+x[2][:, bn2 < bn_threshold])/2.0
        x3[:, bn3 >= bn_threshold] = x[2][:, bn3 >= bn_threshold]
        x3[:, bn3 < bn_threshold] = (x[0][:, bn3 < bn_threshold] + x[1][:, bn3 < bn_threshold]) / 2.0

        return [x1, x2,x3]
class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        #就是大于阈值的那些通道保留，小于阈值的那些通道取另外一个的值
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:,torch.abs( bn1) >= bn_threshold] = x[0][:,  torch.abs( bn1)>= bn_threshold]
        x1[:, torch.abs( bn1) < bn_threshold] = x[1][:, torch.abs( bn1)< bn_threshold]
        x2[:, torch.abs( bn2) >= bn_threshold] = x[1][:, torch.abs( bn2)>= bn_threshold]
        x2[:, torch.abs( bn2) < bn_threshold] = x[0][:, torch.abs( bn2)< bn_threshold]
        # print('bn',bn1 < bn_threshold)

        return [x1, x2]
# class Exchange(nn.Module):
#     def __init__(self):
#         super(Exchange, self).__init__()

#     def forward(self, x, bn, bn_threshold):
#         bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
#         x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
#         x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
#         x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
#         x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
#         x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
#         return [x1, x2]
class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel=2):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(int(num_parallel)):

            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=1, bias=bias))


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))

class DBlock(nn.Module):
    def __init__(self, channel):
        super(DBlock, self).__init__()
        self.dilate1 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=8, padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out
class DBlock_parallel(nn.Module):
    def __init__(self, channel,num_parallel):
        super(DBlock_parallel, self).__init__()
        self.dilate1 = ModuleParallel(nn.Conv2d(
            channel, channel, kernel_size=3, dilation=1, padding=1))
        self.dilate2 = ModuleParallel(nn.Conv2d(
            channel, channel, kernel_size=3, dilation=2, padding=2))
        self.dilate3 = ModuleParallel(nn.Conv2d(
            channel, channel, kernel_size=3, dilation=4, padding=4))
        self.dilate4 = ModuleParallel(nn.Conv2d(
            channel, channel, kernel_size=3, dilation=8, padding=8))
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        self.num_parallel=num_parallel
    def forward(self, x):
        dilate1_out = self.relu(self.dilate1(x))
        dilate2_out = self.relu(self.dilate2(dilate1_out))
        dilate3_out = self.relu(self.dilate3(dilate2_out))
        dilate4_out = self.relu(self.dilate4(dilate3_out))
        out = [x[l] + dilate1_out[l] + dilate2_out[l] + dilate3_out[l] + dilate4_out[l] for l in range(self.num_parallel)]

        return out

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=True, bias=False):
        super(ResidualBlock, self).__init__()
        dim_out = planes
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3),
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        if downsample == True:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(planes),
            )
        elif isinstance(downsample, nn.Module):
            self.downsample = downsample
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.bn2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock1DConv(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bias=False):
        super(BasicBlock1DConv, self).__init__()
        dim_out = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3),
                               stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(1, 9),
                                 stride=1, padding=(0, 4), bias=bias)
        self.conv2_2 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(9, 1),
                                 stride=1, padding=(4, 0), bias=bias)
        self.conv2_3 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(9, 1),
                                 stride=1, padding=(4, 0), bias=bias)
        self.conv2_4 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(1, 9),
                                 stride=1, padding=(0, 4), bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x0 = self.conv2(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(self.h_transform(x))
        x3 = self.inv_h_transform(x3)
        x4 = self.conv2_4(self.v_transform(x))
        x4 = self.inv_v_transform(x4)

        x = x0 + torch.cat((x1, x2, x3, x4), 1)
        out = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

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


class DecoderBlock_parallel(nn.Module):
    def __init__(self, in_channels, n_filters,num_parallel):
        super(DecoderBlock_parallel, self).__init__()

        self.conv1 = conv1x1(in_channels, in_channels // 4, 1)
        self.norm1 = BatchNorm2dParallel(in_channels // 4, num_parallel)
        self.relu1 =  ModuleParallel(nn.ReLU(inplace=True))
        self.deconv2 = ModuleParallel(nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        ))
        self.norm2 = BatchNorm2dParallel(in_channels // 4, num_parallel)
        self.relu2 = ModuleParallel(nn.ReLU(inplace=True))
        self.conv3 = conv1x1(in_channels // 4, n_filters, 1)
        self.norm3 = BatchNorm2dParallel(n_filters, num_parallel)
        self.relu3 = ModuleParallel(nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x



class DecoderBlock_parallel_exchange(nn.Module):
    def __init__(self, in_channels, n_filters,num_parallel,bn_threshold):
        super(DecoderBlock_parallel_exchange, self).__init__()

        self.conv1 = conv1x1(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm2dParallel(in_channels // 4, num_parallel)
        self.relu1 =  ModuleParallel(nn.ReLU(inplace=True))
        self.deconv2 = ModuleParallel(nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        ))
        self.bn2 = BatchNorm2dParallel(in_channels // 4, num_parallel)
        self.relu2 = ModuleParallel(nn.ReLU(inplace=True))
        self.conv3 = conv1x1(in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm2dParallel(n_filters, num_parallel)
        self.relu3 = ModuleParallel(nn.ReLU(inplace=True))
        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        if len(x) > 1:
            x = self.exchange(x, self.bn2_list, self.bn_threshold)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x
BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d
class SpatialGCN(nn.Module):
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


class DualGCN(nn.Module):
    """
        Feature GCN with coordinate GCN
    """
    def __init__(self, planes, ratio=4):
        super(DualGCN, self).__init__()

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

        # cat or sum, nearly the same results
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

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



class SEfuse(torch.nn.Module):  # Dual Enhancement Module
    def __init__(self, in_planes, out_planes, reduction=16, bn_momentum=0.0003):
        self.init__ = super(SEfuse, self).__init__()
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

        return rgb_out, hha_out


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

    def forward(self, x):
        rgb_info,add_info=x
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
import torch
from torch import nn


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0.5)
            self.m_conv.register_full_backward_hook(self._set_lr)
    #论文中可变形卷积的两个学习量的学习率设置的是当前层的0.1，初始化值分别为0和0.5
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        
        offset = self.p_conv(x[1])
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x[1]))
        x=x[0]
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class RefUnet(nn.Module):
    def __init__(self,in_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,in_ch,3,padding=1)

        self.conv1 = nn.Conv2d(in_ch,16,3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####
        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(64+32,32,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(32)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(32+16,16,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(16)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(16,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx))) #scale:1
        hx = self.pool1(hx1) #scale:1/2

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2) #scale:1/4

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3) #scale:1/8

        hx4 = self.relu4(self.bn4(self.conv4(hx)))

        hx = self.upscore2(hx4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual






