import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d
device = torch.device("cuda:0")


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class Sobel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Sobel, self).__init__()
        kernel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        kernel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        kernel_x = torch.FloatTensor(kernel_x).expand(out_channel, in_channel, 3, 3)
        kernel_x = kernel_x.type(torch.cuda.FloatTensor)
        kernel_y = torch.cuda.FloatTensor(kernel_y).expand(out_channel, in_channel, 3, 3)
        kernel_y = kernel_y.type(torch.cuda.FloatTensor)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).clone()
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).clone()
        self.softmax = nn.Softmax()

    def forward(self, x):
        b, c, h, w = x.size()
        sobel_x = F.conv2d(x, self.weight_x, stride=1, padding=1)
        sobel_x = torch.abs(sobel_x)
        sobel_y = F.conv2d(x, self.weight_y, stride=1, padding=1)
        sobel_y = torch.abs(sobel_y)
        if c == 1:
            sobel_x = sobel_x.view(b, h, -1)
            sobel_y = sobel_y.view(b, h, -1).permute(0, 2, 1)
        else:
            sobel_x = sobel_x.view(b, c, -1)
            sobel_y = sobel_y.view(b, c, -1).permute(0, 2, 1)
        sobel_A = torch.bmm(sobel_x, sobel_y)
        sobel_A = self.softmax(sobel_A)
        return sobel_A


class GCNSpatial(nn.Module):
    def __init__(self, channels):
        super(GCNSpatial, self).__init__()
        self.sobel = Sobel(channels, channels)
        self.fc1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def normalize(self, A):
        b, c, im = A.size()
        out = np.array([])
        for i in range(b):
            A1 = A[i].to(device="cpu")
            I = torch.eye(c, im)
            A1 = A1 + I
            # degree matrix
            d = A1.sum(1)
            # D = D^-1/2
            D = torch.diag(torch.pow(d, -0.5))
            new_A = D.mm(A1).mm(D).detach().numpy()
            out = np.append(out, new_A)
        out = out.reshape(b, c, im)
        normalize_A = torch.from_numpy(out)
        normalize_A = normalize_A.type(torch.cuda.FloatTensor)
        return normalize_A

    def forward(self, x):
        b, c, h, w = x.size()
        A = self.sobel(x)
        A = self.normalize(A)
        x = x.view(b, c, -1)
        x = F.relu(self.fc1(A.bmm(x)))
        x = F.relu(self.fc2(A.bmm(x)))
        x = self.fc3(A.bmm(x))
        out = x.view(b, c, h, w)
        return out


class GCNChannel(nn.Module):
    def __init__(self, channels):
        super(GCNChannel, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.sobel = Sobel(1, 1)
        self.fc1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    # def sobel_channel(self,x):
    #     b,c,h,w = x.size()
    #     sobel = Sobel(h*w,h*w)
    #     return sobel

    def pre(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = x.view(b, 1, h * w, c)
        return x

    def normalize(self, A):
        b, c, im = A.size()
        out = np.array([])
        for i in range(b):
            # A = A = I
            A1 = A[i].to(device="cpu")
            I = torch.eye(c, im)
            A1 = A1 + I
            # degree matrix
            d = A1.sum(1)
            # D = D^-1/2
            D = torch.diag(torch.pow(d, -0.5))
            new_A = D.mm(A1).mm(D).detach().numpy()
            out = np.append(out, new_A)
        out = out.reshape(b, c, im)
        normalize_A = torch.from_numpy(out)
        normalize_A = normalize_A.type(torch.cuda.FloatTensor)
        return normalize_A

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.input(x)
        b, c, h1, w1 = x.size()
        x = self.pre(x)
        # A = self.sobel_channel(x)
        A = self.sobel(x)
        A = self.normalize(A)
        x = x.view(b, -1, c)
        x = F.relu(self.fc1(A.bmm(x).permute(0, 2, 1))).permute(0, 2, 1)
        x = F.relu(self.fc2(A.bmm(x).permute(0, 2, 1))).permute(0, 2, 1)
        x = self.fc3(A.bmm(x).permute(0, 2, 1))
        out = x.view(b, c, h1, w1)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        return out


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        # ConvTranspose
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_copy, x, interpolate=True):
        out = self.up(x)
        if interpolate:
            # Iterative filling, in order to obtain better results
            out = F.interpolate(out, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True
                                )
        else:
            # Different filling volume
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            out = F.pad(out, (diffX // 2, diffX - diffX // 2, diffY, diffY - diffY // 2))
        # Splicing
        out = torch.cat([x_copy, out], dim=1)
        out_conv = self.up_conv(out)
        return out_conv


class TwofoldGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(TwofoldGCN, self).__init__()
        # Depthwise convolution # for spatial feature extraction
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            BatchNorm2d(out_channels)
        )
        # GCN Spatial #
        self.spatial_in = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.gcn_s = GCNSpatial(out_channels // 2)
        self.conv_s = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channels // 2)
        )

        # Pointwise convolution # for channel feature extraction
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            BatchNorm2d(out_channels)
        )
        # GCN Channel #
        self.channel_in = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, stride=1),
            BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.gcn_c = GCNChannel(out_channels // 2)
        self.conv_c = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channels // 2)
        )
        # output
        self.combine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels)
        )
        self.output = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channels),
            nn.ReLU(out_channels),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, bias=True)
            )

    def forward(self, x):
        # GCN_Spatial
        x_spatial_in = self.depth_conv(x)
        x_spatial_in = self.spatial_in(x_spatial_in)
        x_spatial = self.gcn_s(x_spatial_in)
        x_spatial = x_spatial_in + x_spatial
        # GCN_Channel
        x_channel_in = self.channel_conv(x)
        x_channel_in = self.channel_in(x_channel_in)
        x_channel = self.gcn_c(x_channel_in)
        x_channel = x_channel_in + x_channel
        # out
        out = torch.cat((x_spatial, x_channel), 1) + x
        out = self.combine(out)
        out = self.output(out)
        return out

