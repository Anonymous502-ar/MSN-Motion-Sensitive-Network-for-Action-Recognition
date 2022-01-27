import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import math
import torch.nn.functional as F
from ops.base_module import *
# from ops.classifier import classifier

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Gate(nn.Module):
    def __init__(self, gate_channels):
        super(Gate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels),
            nn.Sigmoid(),
        )
        self.channel_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_max_pooling = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        channel_avg_pool = self.channel_avg_pooling(x)
        channel_avg_pool = self.mlp(channel_avg_pool)
        channel_max_pool = self.channel_max_pooling(x)
        channel_max_pool = self.mlp(channel_max_pool)

        channel_att_sum = (channel_avg_pool + channel_max_pool) * 0.5
        # scale = tor ch.sigmoid(channel_att_sum).view(x.size(0), -1, 1, 1, 1).expand_as(x)
        # scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        scale = channel_att_sum.unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class MSN_Net(nn.Module):

    def __init__(self,resnet_model,resnet_model1,apha,belta, segmentation = 8):
        super(MSN_Net, self).__init__()

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)
        self.segmentation = segmentation
        # implement conv1_5 and inflate weight
        self.conv1_temp = list(resnet_model1.children())[0]
        params = [x.clone() for x in self.conv1_temp.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * 4,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv1_5 = nn.Sequential(nn.Conv2d(12,64,kernel_size=7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.conv1_5[0].weight.data = new_kernels

        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resnext_layer1 =nn.Sequential(*list(resnet_model1.children())[4])
        # self.resnext_layer2 = nn.Sequential(*list(resnet_model1.children())[5])
        # self.resnext_layer3 = nn.Sequential(*list(resnet_model1.children())[6])
        # self.resnext_layer4 = nn.Sequential(*list(resnet_model1.children())[7])

        #self.gate_1 = Gate(64)
        #self.gate_2 = Gate(256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avg_diff = nn.AvgPool2d(kernel_size=2,stride=2)
        # self.norm = nn.BatchNorm2d(256)
        self.fc = list(resnet_model.children())[8]
        self.apha = apha
        self.belta = belta
        # self.classifier = classifier(class_num=174)
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.conv_2 = nn.Conv2d(16, 64, kernel_size = 5, stride=1, padding=2)
        self.conv_diff_1 = nn.Sequential(
            nn.Conv2d(12,64,kernel_size=7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True)
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        # self.conv_diff_3 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        # )
    def pyrimidModule(self, x):
        '''
        x : b, frames * 3, h, w
        '''
        x = x.squeeze()
        x1, x2, x3, x4, x5 = x[:, 0:3, :, :], x[:, 3:6, :, :], x[:, 6:9, :, :], x[:, 9:12, :, :], x[:, 12:15, :, :]
        x_diff_1 = self.conv_diff_1 (self.maxpool_diff(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],1))) # 56*56
        # print(x_diff_1.size())

        x1 = self.conv_1(x1) #112 *112
        x3 = self.conv_1(x3)
        x5 = self.conv_1(x5)
        x_diff_2 = self.conv_diff_2(self.maxpool_diff(torch.cat([x5-x3, x3-x1], dim=1))) # #56 * 56
        # print(x_diff_2.size())

        x1 = self.conv_2(x1) #56*56
        x5 = self.conv_2(x5)
        x_diff_3 = self.maxpool_diff(x5-x1)
        # print(x_diff_3.size())
        res = x_diff_1 + x_diff_2 + x_diff_3

        return res

    def forward(self, raw_x):
        # x1, x2, x3, x4, x5 = x[:,0:3,:,:], x[:,3:6,:,:], x[:,6:9,:,:], x[:,9:12,:,:], x[:,12:15,:,:]
        # x_c5 = self.conv1_5(self.avg_diff(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],1).view(-1,12,x2.size()[2],x2.size()[3])))
        # x_diff = self.maxpool_diff(1.0/1.0*x_c5)
        x = raw_x[:,6:9,:,:]
        bt, c, h, w = raw_x.size()
        t_list = list(torch.split(raw_x.view(-1, self.segmentation, c, h, w), 1, dim=1))

        res_list = []
        for t in t_list:
            res_list.append(self.pyrimidModule(t).unsqueeze(dim=1))
        x_diff = torch.cat(res_list, dim=1).view(bt, 64, h//4, w//4)

        temp_out_diff1 = x_diff


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #fusion layer1
        x = self.maxpool(x)
        temp_out_diff1 = F.interpolate(temp_out_diff1, x.size()[2:])
        x = self.apha*x + self.belta * temp_out_diff1
        #fusion layer2
        x = self.layer1_bak(x)
        x_diff = self.resnext_layer1(x_diff)
        x = 0.75 * x + 0.25 * x_diff

        # x = self.norm(x)
        x = self.layer2_bak(x)

        x = self.layer3_bak(x)

        x = self.layer4_bak(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

def msn_net(base_model=None,num_segments=8,pretrained=True, **kwargs):
    if("50" in base_model):
        resnet_model = fbresnet50(num_segments, pretrained)
        resnet_model1 = fbresnet50(num_segments, pretrained)
    else:
        resnet_model = fbresnet101(num_segments, pretrained)
        resnet_model1 = fbresnet101(num_segments, pretrained)

    if(num_segments is 8):
        model = MSN_Net(resnet_model,resnet_model1,apha=0.5,belta=0.5)
    else:
        model = MSN_Net(resnet_model,resnet_model1,apha=0.75,belta=0.25)
    return model


