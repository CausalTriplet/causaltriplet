#!/usr/bin/env python3

import torch
from torch.nn import functional

import pdb


def sparse_criterion_with_label(z1, z2, blk):
    '''
        z1, z2: feature vectors in the shape of [B, D]
        blk: mask of the intervened block in the shape of [B, K]
    '''
    ratio = z1.shape[1] // blk.shape[1]
    extra = z1.shape[1] % blk.shape[1]
    blk_action = torch.cat([torch.repeat_interleave(blk, ratio, dim=1), torch.repeat_interleave(blk[:, -1:], extra, dim=1)], dim=1)
    blk_invariant = 1 - blk_action
    loss = functional.mse_loss(z1 * blk_invariant, z2 * blk_invariant)

    return loss
