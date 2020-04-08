# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2, os
from torchvision.transforms import ToPILImage
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.alexnet import alexnet
from .backbones.vgg2 import vgg11_bn
from .backbones.densenet import densenet121

# torch.cuda.manual_seed_all(1)
# torch.manual_seed(1)
# np.random.seed(1)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def my_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.333)
        #nn.init.normal_(m.weight, mean=0.3, std=0.1)
        nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)

def norm(x):
    x = 1. * x / (torch.norm(x, 2, -1, keepdim=True).expand_as(x) + 1e-12)
    return x

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()

        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)


        self.attention = nn.Sequential(
            nn.Conv2d(2048, 128, kernel_size=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2048, kernel_size=1,padding=0),
            nn.Sigmoid(),)
        self.attention.apply(weights_init_kaiming)

        self.generate1 = nn.Linear(3, 1)
        self.generate1.apply(my_weights_init)
        self.generate2 = nn.Linear(1, 3)
        self.generate2.apply(my_weights_init)

    def forward(self, x):
        B, C, H, W = x.shape
        indx = np.arange(B//2)*2

        if self.training:
            gray1 = F.relu(self.generate1(x[indx].view(B//2, C, -1).permute(0, 2, 1).contiguous().view(-1, C)))
            gray1 = self.generate2(gray1).view(B//2, -1, 3).permute(0,2,1).contiguous().view(B//2,3,H,W)
            x = torch.cat((x, gray1))

        global_feat = self.gap(self.base(x))

        mask = self.attention(global_feat)
        pos = global_feat * (1 + mask)
        neg = global_feat * (1 - mask)

        pos = pos.view(x.shape[0], -1)
        neg = neg.view(x.shape[0], -1)


        if self.training:
            cls_score_pos = self.classifier(self.bottleneck(pos))
            cls_score_neg = self.classifier(self.bottleneck(neg))
            return cls_score_pos, cls_score_neg, pos, neg


        else:
            if self.neck_feat == 'after':
                return self.bottleneck(pos), self.bottleneck(neg)
            else:
                return pos, neg







    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self[i].copy_(param_dict[i])
            #self.state_dict()[i].copy_(param_dict[i])
