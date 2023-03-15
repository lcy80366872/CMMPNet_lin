import torch
import torch.nn as nn

class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        #groups：将输入输出通道进行对应分组，groups=dim代表一个输入通道对应一个输出通道，即dw卷积
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
class MSCA(nn.Module):
    def __init__(self, dim):
        super(MSCA,self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u
class Attention(nn.Module):
    def __init__(self, chanel):
        super(Attention,self).__init__()
        self.activate=nn.GELU()
        self.conv1=nn.Conv2d(chanel,chanel,1)
        self.MSCA=MSCA(chanel)
        self.conv2=nn.Conv2d(chanel,chanel,1)
    def forward(self,x):
        shorcut = x.clone()
        con1=self.conv1(x)
        activate=self.activate(con1)
        msca=self.MSCA(activate)
        out=self.conv2(msca)
        return out+shorcut
class FFN(nn.Module):
    def __init__(self, chanel):
        super(FFN,self).__init__()
        self.activate = nn.GELU()
        self.conv1 = nn.Conv2d(chanel, chanel, 1)
        self.dwconv = DWConv(chanel)
        self.conv2 = nn.Conv2d(chanel, chanel, 1)
    def forward(self,x):
        con1=self.conv1(x)
        dw=self.dwconv(con1)
        activate=self.activate(dw)
        out = self.conv2(activate)
        return  out

class MSCAN(nn.Module):
    def __init__(self, chanel):
        super(MSCAN,self).__init__()
        self.BN = nn.BatchNorm2d(chanel)
        self.atten=Attention(chanel)
        self.ffn=FFN(chanel)
    def forward(self,x):
        shortcut1=x.clone()
        bn1=self.BN(x)
        atten=self.atten(bn1)
        out_middle=shortcut1+atten
        shortcut2=out_middle.clone()
        bn2=self.BN(out_middle)
        ffn=self.ffn(bn2)
        out=shortcut2+ffn
        return out

class down_sampling(nn.Module):
    def __init__(self, in_chanel,out_chanel):
        super(down_sampling,self).__init__()
        self.conv=nn.Conv2d(in_chanel,out_chanel,kernel_size=3,stride=2,padding=1)
        self.norm=nn.BatchNorm2d(out_chanel)
    def forward(self,x):
        conv=self.conv(x)
        norm=self.norm(conv)
        return  norm
class encoderblock(nn.Module):
    def __init__(self, in_chanel,out_chanel):
        super(encoderblock,self).__init__()
        self.down_sampling=down_sampling(in_chanel,out_chanel)
        self.mscan=MSCAN(out_chanel)
    def forward(self,x):

        down=self.down_sampling(x)
        mscan=self.mscan(down)
        out=mscan
        return out



