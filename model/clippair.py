import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import clip
import logging

import pdb


def convert_models_to_fp32(model):
    # from https://github.com/openai/CLIP/issues/57
    for p in model.parameters():
        p.data = p.data.float()
        if p.requires_grad:
            p.grad.data = p.grad.data.float()


def convert_models_to_mix(model):
    clip.model.convert_weights(model)


class ClipPair(nn.Module):
    def __init__(self, feat_dim=64, num_class=10, linear=False):
        super(ClipPair, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder, _ = clip.load("ViT-B/32", device=device)

        # freeze all layers but the last fc
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        dim_in = 512

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

            self.nonlinear = 'subcon'

            logging.info(f'Nonlinear relation module: {self.nonlinear}')

        self.classifier = nn.Linear(feat_dim, num_class)

    def forward(self, x1, x2):
        f1 = self.encoder.encode_image(x1)
        f2 = self.encoder.encode_image(x2)
        # action embedding
        p1 = self.proj(f1)
        p2 = self.proj(f2)
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
