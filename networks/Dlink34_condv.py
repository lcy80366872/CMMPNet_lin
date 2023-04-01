import torch
from torch import Tensor
import torch.nn as nn
from networks.CondConv import CondConv
from typing import Type, Any, Callable, Union, List, Optional
from .basic_blocks import *
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        downsample1: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        condconv = False
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.in_chanel=inplanes
        self.conv1 =conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv1_g = conv3x3(inplanes, planes, stride)
        self.bn1_g = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.condconv=condconv
        if condconv == False:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = CondConv(planes, planes, kernel_size=3, stride=1,
                                  padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv2_g = conv3x3(planes, planes)
        self.bn2_g = norm_layer(planes)
        self.downsample = downsample
        self.downsample1 = downsample1
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        a = self.in_chanel
        x = input[:, :a, :, :]
        g = input[:, a:, :, :]
        # print('x:',x.shape)
        # print('g:', g.shape)
        residual = x
        residual_g = g
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        g = self.conv1_g(g)
        g = self.bn1_g(g)
        g = self.relu(g)
        # print('out:',out.shape)
        # print('g:', g.shape)
        # print('out1',out.shape)
        if self.condconv == False:
            out = self.conv2(out)

        else:
            out = self.conv2(out, g)
        # print('out2', out.shape)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        print('identity', residual.shape)
        out += residual
        out = self.relu(out)

        out_g = self.conv2_g(g)
        out_g = self.bn2_g(out_g)
        if self.downsample1 is not None:
            residual_g = self.downsample1(residual_g)
        out_g += residual_g
        out_g = self.relu(out_g)

        out = torch.cat((out, out_g), 1)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        filters = [64, 128, 256, 512]
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.conv1_g = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2,
                                 padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], condconv=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, condconv=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, condconv=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, condconv=False)

        self.dblock = DBlock(filters[3])
        self.dblock_add = DBlock(filters[3])
        # decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.decoder4_add = DecoderBlock(filters[3], filters[2])
        self.decoder3_add = DecoderBlock(filters[2], filters[1])
        self.decoder2_add = DecoderBlock(filters[1], filters[0])
        self.decoder1_add = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1_add = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1_add = nonlinearity
        self.finalconv2_add = nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1)
        self.finalrelu2_add = nonlinearity

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(filters[0] // 2, filters[0] // 2, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv = nn.Conv2d(filters[0], num_classes, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False,condconv=False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        downsample1 = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            downsample1= nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, downsample1,self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, inputs: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = inputs[:, :3, :, :]
        g = inputs[:, 3:, :, :]
        # print('img:',x.shape)
        # img
        ##stem layer
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out_g = self.relu(self.bn1(self.conv1_g(g)))
        out_g = self.maxpool(out_g)

        # out = out, out_g
        out = torch.cat((out, out_g), 1)

        ##layers:
        x_1 = self.layer1(out)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        x_e1 = x_1[:, :64, :, :]
        g_e1 = x_1[:, 64:, :, :]
        x_e2 = x_2[:, :128, :, :]
        g_e2 = x_2[:, 128:, :, :]

        x_e3 = x_3[:, :256, :, :]
        g_e3 = x_3[:, 256:, :, :]

        x_e4 = x_4[:, :512, :, :]
        g_e4 = x_4[:, 512:, :, :]

        g_c = self.dblock(g_e4)
        x_c = self.dblock(x_e4)
        # decoder
        x_d4 = self.decoder4(x_c) + x_e3
        x_d3 = self.decoder3(x_d4) + x_e2
        x_d2 = self.decoder2(x_d3) + x_e1
        x_d1 = self.decoder1(x_d2)

        g_d4 = self.decoder4_add(g_c) + g_e3
        g_d3 = self.decoder3_add(g_d4) + g_e2
        g_d2 = self.decoder2_add(g_d3) + g_e1
        g_d1 = self.decoder1_add(g_d2)

        x_out = self.finalrelu1(self.finaldeconv1(x_d1))
        x_out = self.finalrelu2(self.finalconv2(x_out))
        g_out = self.finalrelu1(self.finaldeconv1(g_d1))
        g_out = self.finalrelu2(self.finalconv2(g_out))
        out = self.finalconv(torch.cat((x_out, g_out), 1))
        # out=self.finalconv(x_out)
        out = torch.sigmoid(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
def DinkNet34_CMMPNet():
    model = ResNet(BasicBlock, [3, 4, 6, 3], 1)
    return model

def ResNet50(num_classes=1000):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=1000):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=1000):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)

def DinkNet34_CMMPNet(num_classes=1):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    return model
if __name__ == '__main__':
    input = torch.randn(50, 3, 224, 224)
    resnet50 = ResNet50(1000)
    # resnet101=ResNet101(1000)
    # resnet152=ResNet152(1000)
    out = resnet50(input)
    print(out.shape)

