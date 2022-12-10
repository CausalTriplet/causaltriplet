import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import pdb

import mmcv
import numpy as np
from einops import rearrange
from mmseg.models import EncoderDecoder

import os.path as osp

GROUP_PALETTE = np.loadtxt(osp.join(osp.dirname(osp.abspath(__file__)), 'mask_palette.txt'), dtype=np.uint8)[:, ::-1]


def group_to_map(groups, img, return_onehot=False, rescale=False):
    """
    Args:
        img: [B, C, H, W]

    Returns:
        attn_maps: list[Tensor], attention map of shape [B, H, W, groups]
    """

    attn_maps = []
    with torch.no_grad():
        prev_attn_masks = None
        for idx, attn_dict in enumerate(groups['attn_dicts']):
            if attn_dict is None:
                assert idx == len(groups['attn_dicts']) - 1, 'only last layer can be None'
                continue
            # [B, G, HxW]
            # B: batch size (1), nH: number of heads, G: number of group token
            attn_masks = attn_dict['soft']
            # [B, nH, G, HxW] -> [B, nH, HxW, G]
            attn_masks = rearrange(attn_masks, 'b h g n -> b h n g')
            if prev_attn_masks is None:
                prev_attn_masks = attn_masks
            else:
                prev_attn_masks = prev_attn_masks @ attn_masks
            # [B, nH, HxW, G] -> [B, nH, H, W, G]
            attn_maps.append(resize_attn_map(prev_attn_masks, *img.shape[-2:]))

    for i in range(len(attn_maps)):
        attn_map = attn_maps[i]
        # [B, nh, H, W, G]
        assert attn_map.shape[1] == 1
        # [B, H, W, G]
        attn_map = attn_map.squeeze(1)

        if rescale:
            attn_map = rearrange(attn_map, 'b h w g -> b g h w')
            attn_map = F.interpolate(
                attn_map, size=img.shape[2:], mode='bilinear', align_corners=False)
            attn_map = rearrange(attn_map, 'b g h w -> b h w g')

        if return_onehot:
            # [B, H, W, G]
            attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)

        attn_maps[i] = attn_map

    return attn_maps


def resize_attn_map(attentions, h, w, align_corners=False):
    """

    Args:
        attentions: shape [B, num_head, H*W, groups]
        h:
        w:

    Returns:

        attentions: shape [B, num_head, h, w, groups]


    """
    scale = (h * w // attentions.shape[2])**0.5
    if h > w:
        w_featmap = w // int(np.round(scale))
        h_featmap = attentions.shape[2] // w_featmap
    else:
        h_featmap = h // int(np.round(scale))
        w_featmap = attentions.shape[2] // h_featmap
    assert attentions.shape[
        2] == h_featmap * w_featmap, f'{attentions.shape[2]} = {h_featmap} x {w_featmap}, h={h}, w={w}'

    bs = attentions.shape[0]
    nh = attentions.shape[1]  # number of head
    groups = attentions.shape[3]  # number of group token
    # [bs, nh, h*w, groups] -> [bs*nh, groups, h, w]
    attentions = rearrange(
        attentions, 'bs nh (h w) c -> (bs nh) c h w', bs=bs, nh=nh, h=h_featmap, w=w_featmap, c=groups)
    attentions = F.interpolate(attentions, size=(h, w), mode='bilinear', align_corners=align_corners)
    #  [bs*nh, groups, h, w] -> [bs, nh, h*w, groups]
    attentions = rearrange(attentions, '(bs nh) c h w -> bs nh h w c', bs=bs, nh=nh, h=h, w=w, c=groups)

    return attentions


def show_mask(img_show, attn_map_list, out_file, vis_mode='input', opacity=0.5):

    num_groups = [attn_map_list[layer_idx].shape[-1] for layer_idx in range(len(attn_map_list))]
    for layer_idx, attn_map in enumerate(attn_map_list):
        if vis_mode == 'first_group' and layer_idx != 0:
            continue
        if vis_mode == 'final_group' and layer_idx != len(attn_map_list) - 1:
            continue
        attn_map = rearrange(attn_map, 'b h w g -> b g h w')
        attn_map = F.interpolate(
            attn_map, size=img_show.shape[:2], mode='bilinear', align_corners=False)
        group_result = attn_map.argmax(dim=1).cpu().numpy()

        # pdb.set_trace()
        # print(group_result)

        if vis_mode == 'all_groups':
            layer_out_file = out_file.replace(
                osp.splitext(out_file)[-1], f'_layer{layer_idx}{osp.splitext(out_file)[-1]}')
        else:
            layer_out_file = out_file

        blend_result(
            img=img_show,
            result=group_result,
            palette=GROUP_PALETTE[sum(num_groups[:layer_idx]):sum(num_groups[:layer_idx + 1])],
            out_file=layer_out_file,
            opacity=opacity)


def blend_result(img, result, palette=None, out_file=None, opacity=0.5, with_bg=False):
    # img = mmcv.imread(img)
    img = img.copy()
    seg = result[0]
    palette = np.array(palette)
    assert palette.shape[1] == 3, palette.shape
    assert len(palette.shape) == 2
    assert 0.0 <= opacity <= 1.0
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    if opacity > 0:
        if with_bg:
            fg_mask = seg != 0
            img[fg_mask] = img[fg_mask] * (1 - opacity) + color_seg[fg_mask] * opacity
        else:
            img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)

    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


class GroupAverage(nn.Module):
    def __init__(self, encoder, feat_dim=64, num_class=10):
        super(GroupAverage, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = encoder

        # freeze all layers but the last fc
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        dim_in = 384

        self.proj = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.ReLU(inplace=True)
            )

        self.pairwise = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True)
            )

        self.classifier = nn.Linear(feat_dim, num_class)

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        # action embedding
        p1 = self.proj(f1)
        p2 = self.proj(f2)
        z = self.pairwise(torch.cat([p1, p2-p1], axis=1))
        # output
        logit = self.classifier(z)
        return logit, z, p1, p2


class GroupDense(nn.Module):
    def __init__(self, encoder, feat_dim=64, num_class=10):
        super(GroupDense, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = encoder

        self.feat_dim = feat_dim

        # freeze all layers but the last fc
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        dim_in = 384

        self.proj = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.ReLU(inplace=True)
            )

        self.pairwise = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True)
            )

        self.classifier = nn.Linear(feat_dim, num_class)

    def forward(self, x1, x2):
        _, g1 = self.encoder(x1, return_feat=True)
        _, g2 = self.encoder(x2, return_feat=True)
        # action
        p1 = self.proj(g1)
        p2 = self.proj(g2)
        # pairwise
        dense = torch.cat([p1.repeat(1, 8, 1), p2.repeat_interleave(8, dim=1)-p1.repeat(1, 8, 1)], axis=-1)
        z = self.pairwise(dense).mean(dim=1)
        # output
        logit = self.classifier(z)
        return logit, z, p1, p2


class GroupMatchTokenMean(nn.Module):
    def __init__(self, encoder, feat_dim=64, num_class=10):
        super(GroupMatchTokenMean, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = encoder
        self.feat_dim = feat_dim

        # freeze all layers but the last fc
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        dim_in = 384

        self.proj = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.ReLU(inplace=True)
            )

        self.pairwise = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True)
            )

        self.classifier = nn.Linear(feat_dim, num_class)

    def forward(self, x1, x2):
        _, g1 = self.encoder(x1, return_feat=True)
        _, g2 = self.encoder(x2, return_feat=True)
        # action
        p1 = self.proj(g1)
        p2 = self.proj(g2)
        z = self.pairwise(torch.cat([p1, p2-p1], axis=-1)).mean(dim=1)
        # output
        logit = self.classifier(z)
        return logit, z, p1, p2


class GroupMatchTokenMax(nn.Module):
    def __init__(self, encoder, feat_dim=64, num_class=10):
        super(GroupMatchTokenMax, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = encoder
        self.feat_dim = feat_dim

        # freeze all layers but the last fc
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        dim_in = 384

        self.proj = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.ReLU(inplace=True)
            )

        self.pairwise = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True)
            )

        self.classifier = nn.Linear(feat_dim, num_class)

    def forward(self, x1, x2):
        _, g1 = self.encoder(x1, return_feat=True)
        _, g2 = self.encoder(x2, return_feat=True)
        # action
        p1 = self.proj(g1)
        p2 = self.proj(g2)
        z = self.pairwise(torch.cat([p1, p2-p1], axis=-1)).max(dim=1)[0]
        # output
        logit = self.classifier(z)
        return logit, z, p1, p2


class GroupMask(EncoderDecoder):

    def __init__(self, model):
        super(EncoderDecoder, self).__init__()
        self.model = model

        # self.register_buffer('text_embedding', text_embedding)
        self.align_corners = False

    def get_attn_maps(self, img, return_onehot=False, rescale=False):
        """
        Args:
            img: [B, C, H, W]

        Returns:
            attn_maps: list[Tensor], attention map of shape [B, H, W, groups]
        """
        results = self.model.img_encoder(img, return_attn=True, as_dict=True)

        attn_maps = []
        with torch.no_grad():
            prev_attn_masks = None
            for idx, attn_dict in enumerate(results['attn_dicts']):
                if attn_dict is None:
                    assert idx == len(results['attn_dicts']) - 1, 'only last layer can be None'
                    continue
                # [B, G, HxW]
                # B: batch size (1), nH: number of heads, G: number of group token
                attn_masks = attn_dict['soft']
                # [B, nH, G, HxW] -> [B, nH, HxW, G]
                attn_masks = rearrange(attn_masks, 'b h g n -> b h n g')
                if prev_attn_masks is None:
                    prev_attn_masks = attn_masks
                else:
                    prev_attn_masks = prev_attn_masks @ attn_masks
                # [B, nH, HxW, G] -> [B, nH, H, W, G]
                attn_maps.append(resize_attn_map(prev_attn_masks, *img.shape[-2:]))

        for i in range(len(attn_maps)):
            attn_map = attn_maps[i]
            # [B, nh, H, W, G]
            assert attn_map.shape[1] == 1
            # [B, H, W, G]
            attn_map = attn_map.squeeze(1)

            if rescale:
                attn_map = rearrange(attn_map, 'b h w g -> b g h w')
                attn_map = F.interpolate(
                    attn_map, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
                attn_map = rearrange(attn_map, 'b g h w -> b h w g')

            if return_onehot:
                # [B, H, W, G]
                attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)

            attn_maps[i] = attn_map

        return attn_maps

    def blend_result(self, img, result, palette=None, out_file=None, opacity=0.5, with_bg=False):
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        if palette is None:
            palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[1] == 3, palette.shape
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        if with_bg:
            fg_mask = seg != 0
            img[fg_mask] = img[fg_mask] * (1 - opacity) + color_seg[fg_mask] * opacity
        else:
            img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)

        if out_file is not None:
            mmcv.imwrite(img, out_file)

        return img

    def show_result(self, img_show, img_tensor, result, out_file, vis_mode='input'):

        attn_map_list = self.get_attn_maps(img_tensor)
        assert len(attn_map_list) in [1, 2]
        # only show 16 groups for the first stage
        # if len(attn_map_list) == 2:
        #     attn_map_list[0] = top_groups(attn_map_list[0], k=16)

        num_groups = [attn_map_list[layer_idx].shape[-1] for layer_idx in range(len(attn_map_list))]
        for layer_idx, attn_map in enumerate(attn_map_list):
            if vis_mode == 'first_group' and layer_idx != 0:
                continue
            if vis_mode == 'final_group' and layer_idx != len(attn_map_list) - 1:
                continue
            attn_map = rearrange(attn_map, 'b h w g -> b g h w')
            attn_map = F.interpolate(
                attn_map, size=img_show.shape[:2], mode='bilinear', align_corners=self.align_corners)
            group_result = attn_map.argmax(dim=1).cpu().numpy()
            if vis_mode == 'all_groups':
                layer_out_file = out_file.replace(
                    osp.splitext(out_file)[-1], f'_layer{layer_idx}{osp.splitext(out_file)[-1]}')
            else:
                layer_out_file = out_file
            self.blend_result(
                img=img_show,
                result=group_result,
                palette=GROUP_PALETTE[sum(num_groups[:layer_idx]):sum(num_groups[:layer_idx + 1])],
                out_file=layer_out_file,
                opacity=0.5)
