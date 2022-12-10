import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import pdb


class Adversary(nn.Module):
    def __init__(self, feat_dim=64, num_class=10):
        super(Adversary, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_class)
        )

    def forward(self, z):
        logit = self.classifier(z)
        return logit
