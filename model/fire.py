"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

import copy

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
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
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class Classifier(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=-1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        return self.classifier(x)


class FgClassifier(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=-1, init_center=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.weight = nn.Parameter(copy.deepcopy(init_center))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x_norm, w)


class AttrAwareLoss(nn.Module):
    def __init__(self, scale=16, epsilon=0.1):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, positive_mask):
        inputs = self.scale * inputs
        identity_mask = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()

        log_probs = self.logsoftmax(inputs)
        mask = (1 - self.epsilon) * identity_mask + self.epsilon / positive_mask.sum(1, keepdim=True) * positive_mask
        loss = (- mask * log_probs).mean(0).sum()
        return loss


class MaxAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_f = self.maxpooling(x)
        avg_f = self.avgpooling(x)
        return torch.cat((max_f, avg_f), 1)


class FIRe(nn.Module):
    def __init__(self, pool_type='avg', last_stride=1, pretrain=True, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self.P_parts = 2
        self.K_times = 1

        resnet = getattr(torchvision.models, 'resnet50')(pretrained=pretrain)
        resnet.layer4[0].downsample[0].stride = (last_stride, last_stride)
        resnet.layer4[0].conv2.stride = (last_stride, last_stride)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        feature_dim = 2048
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'maxavg':
            self.pool = MaxAvgPool2d()
        self.feature_dim = (2 * feature_dim) if pool_type == 'maxavg' else feature_dim

        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.FAR_bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.FAR_bottleneck.bias.requires_grad_(False)
        self.FAR_bottleneck.apply(weights_init_kaiming)
        self.FAR_classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        self.FAR_classifier.apply(weights_init_classifier)

    def forward(self, x, fgid=None):
        B = x.shape[0]
        x = self.backbone(x)
        global_feat = self.pool(x).flatten(1)  # [B, d]
        global_feat_bn = self.bottleneck(global_feat)

        if self.training and fgid is not None:
            part_h = x.shape[2] // self.P_parts
            FAR_parts = []
            for k in range(self.P_parts):
                part = x[:, :, part_h * k: part_h * (k + 1), :]  # [B, d, h', w]
                mu = part.mean(dim=[2, 3], keepdim=True)
                var = part.var(dim=[2, 3], keepdim=True)
                sig = (var + 1e-6).sqrt()
                mu, sig = mu.detach(), sig.detach()  # [B, d, 1, 1]
                id_part = (part - mu) / sig  # [B, d, h, w]

                neg_mask = fgid.expand(B, B).ne(fgid.expand(B, B).t())  # [B, B]
                neg_mask = neg_mask.type(torch.float32)
                sampled_idx = torch.multinomial(neg_mask, num_samples=self.K_times, replacement=False).\
                    transpose(-1, -2).flatten(0)  # [B, K] -> [BK]
                new_mu = mu[sampled_idx]  # [BK, d, 1, 1]
                new_sig = sig[sampled_idx]  # [BK, d, 1, 1]

                id_part = id_part.repeat(self.K_times, 1, 1, 1)
                FAR_part = (id_part * new_sig) + new_mu  # [B, d, h', w]
                FAR_parts.append(FAR_part)
            FAR_feat = torch.concat(FAR_parts, dim=2)  # [B, d, h, w]
            FAR_feat = self.pool(FAR_feat).flatten(1)
            FAR_feat_bn = self.FAR_bottleneck(FAR_feat)
            y_FAR = self.FAR_classifier(FAR_feat_bn)
            return global_feat_bn, y_FAR
        else:
            return global_feat_bn
