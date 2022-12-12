
'''
    file:   Resnet.py

    date:   2018_05_02
    author: zhangxiong(1025679612@qq.com)
    mark:   copied from pytorch sourc code
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np
import math
import torchvision

from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d)

class ResNet(MetaModule):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = MetaConv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False) #
        self.bn1 = MetaBatchNorm2d(64) #
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
       # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = MetaSequential(
                MetaConv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False), # downsample.0.weight
                MetaBatchNorm2d(planes * block.expansion), # downsample.1.weight & downsample.0.bias
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return MetaSequential(*layers)

    def forward(self, x, params=None):
        x = self.conv1(x, params=self.get_subdict(params, 'conv1'))
        x = self.bn1(x, params=self.get_subdict(params, 'bn1'))
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, params=self.get_subdict(params, 'layer1'))
        x = self.layer2(x, params=self.get_subdict(params, 'layer2'))
        x = self.layer3(x, params=self.get_subdict(params, 'layer3'))
        x = self.layer4(x, params=self.get_subdict(params, 'layer4'))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
       # x = self.fc(x)

        return x

class Bottleneck(MetaModule):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # MetaConv2d
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, params = None):
        residual = x

        out = self.conv1(x, params=self.get_subdict(params, 'conv1'))
        out = self.bn1(out, params=self.get_subdict(params, 'bn1'))
        out = self.relu(out)

        out = self.conv2(out, params=self.get_subdict(params, 'conv2'))
        out = self.bn2(out, params=self.get_subdict(params, 'bn2'))
        out = self.relu(out)

        out = self.conv3(out, params=self.get_subdict(params, 'conv3'))
        out = self.bn3(out, params=self.get_subdict(params, 'bn3'))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def copy_parameter_from_resnet50(model, res50_dict):
    cur_state_dict = model.state_dict()
    for name, param in list(res50_dict.items())[0:None]:
        if name not in cur_state_dict: 
            # print('unexpected ', name, ' !')
            continue 
        if isinstance(param, Parameter): 
            param = param.data
        try:
            cur_state_dict[name].copy_(param)
        except:
            # print(name, ' is inconsistent!')
            continue
    # print('copy state dict finished!')

def load_Res50Model():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    copy_parameter_from_resnet50(model, torchvision.models.resnet50(pretrained = True).state_dict())
    return model

if __name__ == '__main__':
    vx = torch.autograd.Variable(torch.from_numpy(np.array([1, 1, 1])))
    vy = torch.autograd.Variable(torch.from_numpy(np.array([2, 2, 2])))
    vz = torch.cat([vx, vy], 0)
    vz[0] = 100
    print(vz)
    print(vx)