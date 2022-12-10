import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import logging

import pdb


class ResPair(nn.Module):
    def __init__(self, feat_dim=64, num_class=10, name='resnet50', linear=False, amin=1e-4):
        super(ResPair, self).__init__()
        model_ft = models.__dict__[name](pretrained=True)

        dim_in = model_ft.fc.in_features

        self.encoder = torch.nn.Sequential(*list(model_ft.children())[:-2])

        # freeze all layers but the last fc
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        # project head
        self.proj = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.ReLU(inplace=True)
        )

        # relation between image pairs
        self.linear = linear
        if linear:
            self.pairwise = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.pairwise = nn.Sequential(
                nn.Linear(feat_dim*2, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True)
            )

            # self.nonlinear = 'concat'
            self.nonlinear = 'subcon'
            # self.nonlinear = 'subsum'

            logging.info(f'Nonlinear relation module: {self.nonlinear}')

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feat_dim, num_class)

        self.maskmin = amin
        logging.info(f'min mask value: {self.maskmin}')

    def forward(self, x1, x2, s1=None, s2=None):
        '''
            images: x1, x2
            masks: s1, s2
        '''
        # feature map
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        # feature vector
        if (s1 is not None) and (s2 is not None):
            assert self.maskmin > 0
            # min value outside instance mask
            s1 = torch.clamp(s1, min=self.maskmin)
            s2 = torch.clamp(s2, min=self.maskmin)
            # average pooling over the instance mask
            h1 = (f1 * s1).sum((2, 3)) / s1.sum((2, 3))
            h2 = (f2 * s2).sum((2, 3)) / s2.sum((2, 3))
        else:
            h1 = self.avgpool(f1).flatten(1)
            h2 = self.avgpool(f2).flatten(1)
        # action embedding
        p1 = self.proj(h1)
        p2 = self.proj(h2)
        if self.linear:
            z = self.pairwise(p2-p1)
        else:
            if self.nonlinear == 'concat':
                z = self.pairwise(torch.cat([p1, p2], axis=1))
            elif self.nonlinear == 'subcon':
                z = self.pairwise(torch.cat([p1, p2-p1], axis=1))
            elif self.nonlinear == 'subsum':
                z = self.pairwise(torch.cat([p2+p1, p2-p1], axis=1))
            else:
                pass

        logit = self.classifier(z)
        return logit, z, p1, p2
