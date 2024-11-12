import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
import torchvision.models as models
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # You can choose other variants like resnet18, resnet34, resnet101, etc.

    def forward(self, x):
        # Extract features from the ResNet backbone
        features = self.resnet.conv1(x)
        features = self.resnet.bn1(features)
        features = self.resnet.relu(features)
        features = self.resnet.maxpool(features)

        features = self.resnet.layer1(features)
        features = self.resnet.layer2(features)
        features = self.resnet.layer3(features)
        features = self.resnet.layer4(features)

        return features

class BADAAM(nn.Module):
    def __init__(self, num_res=4):
        super(BADAAM, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.resnet = ResNetBackbone()

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        self.attention_module = EncoderSelfAttention(512,64,64,n_module =3)

    def forward(self, x):

        # print(f"the shape of {x.shape}")
        y = self.resnet(x)
        # print(f"the shape of resnet = {y.shape}")
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x)
        # print(f"the output of x_ {x_.shape}")
        res1 = self.Encoder[0](x_)
        # print(f"the output of res1 {res1.shape}")
        # 128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # print(f"the output of res2 {res2.shape}")
        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)


        query = z.permute(0,2,3,1)
        query = F.interpolate(query,size=[z.shape[-1],y.shape[1]],mode='bilinear')
        key_value = y.permute(0,2,3,1)
        key_value = key_value.view(key_value.shape[0],-1,key_value.shape[-1])
        b,h,w,dim = query.shape
        query = query.view(b,-1,dim)

    #    transformer block here
        # print(f"the shape of query = {query.shape} key_value = {key_value.shape}")
        query = self.attention_module(query,key_value)
        reshaped_query = rearrange(query, 'b (h1 w1) d -> b h1 w1 d', h1=h ,w1=w)
        
        reshaped_query = F.interpolate(reshaped_query,size=[z.shape[-1],z.shape[1]],mode='bilinear')        
        reshaped_query = reshaped_query.permute(0,3,1,2)

        z = reshaped_query + z
        # z= self.reluact(z)
        # print(f"the shaped of z={z.shape}")


        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128
        z = self.feat_extract[3](z)


        outputs.append(z_+x_4)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)

        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)


        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        # print(f"the shape of z at third = {z.shape} and x shape is {x.shape}")

        # transformer block here
        outputs.append(z+x)


        return outputs


def build_net():
    return BADAAM()
