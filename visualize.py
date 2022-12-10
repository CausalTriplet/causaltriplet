#!/usr/bin/env python3

import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import pickle
from torch import Tensor

import pdb


def show_split(verb_noun_iid, verb_noun_ood, figname='fig/split.png'):

    verb_noun_all = {**verb_noun_iid, **verb_noun_ood}
    verb_noun_all = {k: v for k, v in sorted(verb_noun_all.items(), key=lambda item: item[0][0]+'_'+item[0][1])}

    # dict to matrix
    cnt_verb, cnt_noun = 0, 0
    dict_verb, dict_noun = dict(), dict()

    for key, value in verb_noun_all.items():
        verb, noun = key[0], key[1]
        if verb not in dict_verb:
            dict_verb[verb] = cnt_verb
            cnt_verb += 1
        if noun not in dict_noun:
            dict_noun[noun] = cnt_noun
            cnt_noun += 1

    mat_iid = np.zeros([cnt_verb, cnt_noun])
    mat_ood = np.zeros([cnt_verb, cnt_noun])

    for key, value in verb_noun_iid.items():
        idx_verb, idx_noun = dict_verb[key[0]], dict_noun[key[1]]
        mat_iid[idx_verb][idx_noun] = len(value)

    for key, value in verb_noun_ood.items():
        idx_verb, idx_noun = dict_verb[key[0]], dict_noun[key[1]]
        mat_ood[idx_verb][idx_noun] = len(value)

    verblist = dict_verb.keys()
    nounlist = dict_noun.keys()

    mat_split = mat_iid - mat_ood

    plt.rcParams["figure.figsize"] = (cnt_noun, cnt_verb)

    rdgn = sns.diverging_palette(h_neg=10, h_pos=240, sep=1, as_cmap=True)
    ax = sns.heatmap(mat_split, cmap=rdgn, vmin=-3, vmax=3, square=True, cbar=False, xticklabels=nounlist, yticklabels=verblist)

    ntick = max(len(verblist), len(nounlist))
    fs = 13 - 0.125 * ntick

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='center', fontsize=fs)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs)
    plt.xlabel('object class', fontsize=fs)  # x-axis label with fontsize 15
    plt.ylabel('action class', fontsize=fs)  # y-axis label with fontsize 15

    plt.savefig(figname, bbox_inches='tight', dpi=600)


def show_pair(first, second, verb, noun, suffix=None, savedir=None, figname=None):

    plt.rcParams["figure.figsize"] = (14, 12)

    fig, axarr = plt.subplots(1, 2)
    fig.subplots_adjust(wspace=0.02, hspace=0.0)

    axarr[0].imshow(first)
    axarr[0].axis('off')

    axarr[1].imshow(second)
    axarr[1].axis('off')

    fig.suptitle(f'class: ({verb}, {noun})',
                 x=0.5,
                 y=0.65,
                 fontsize=16)

    # plt.show()
    if figname is None:
        figname = f'pair_{verb}_{noun}'
    fig.savefig(savedir + figname + suffix + '.png', bbox_inches='tight')
    plt.close()


def show_stat(dict_stat, figname):

    plt.clf()
    plt.rcParams["figure.figsize"] = (8, 8)

    dict_count = []
    dict_prob = []
    total = sum(dict_stat.values())
    for key, cnt in dict_stat.items():
        dict_count.append([key, cnt])
        dict_prob.append([key, cnt/total])
    df_cnt = pd.DataFrame(dict_count, columns=['class', 'count'])
    df_prob = pd.DataFrame(dict_prob, columns=['class', 'prob'])

    # sns_bar = sns.barplot(data=dict_prob, x="class", y='prob')
    # sns_bar.set(ylim=(0, 0.5))
    # sns_bar.set_xticklabels(sns_bar.get_xticklabels(), rotation=30, horizontalalignment='right')

    # fig_cnt = sns_bar.get_figure()
    # fig_cnt.savefig(figname+'_cnt.png')

    plt.clf()

    sns_bar = sns.barplot(data=df_cnt, x="class", y='count')
    sns_bar.set_xticklabels(sns_bar.get_xticklabels(), rotation=30, horizontalalignment='right')

    fig_cnt = sns_bar.get_figure()
    fig_cnt.savefig(figname+'_cnt.png')


def show_tsne(feat_iid, noun_iid, verb_iid, feat_ood, noun_ood, verb_ood, classes_noun, classes_verb, prefix):
    feat = torch.cat([feat_iid, feat_ood])
    embedding = TSNE(n_components=2, init='pca', learning_rate='auto').fit_transform(feat)
    tx, ty = embedding[:, 0], embedding[:, 1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    xmin, xmax = tx.min(), tx.max()
    ymin, ymax = ty.min(), ty.max()

    tx_iid = tx[:feat_iid.size(0)]
    ty_iid = ty[:feat_iid.size(0)]
    tx_ood = tx[feat_iid.size(0):]
    ty_ood = ty[feat_iid.size(0):]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(10, 10)

    # iid verb
    ax = fig.add_subplot(2, 2, 1)

    num_class = len(np.unique(verb_iid))
    if num_class == 10:
        cmap = 'tab10'
    elif num_class == 2:
        cmap = colors.ListedColormap(['red', 'blue'])
    else:
        cmap = 'rainbow'

    for i in range(len(classes_verb)):
        cond = verb_iid == i
        ax.scatter(tx_iid[cond], ty_iid[cond], cmap=cmap, s=4.0, alpha=1.0, label=classes_verb[i])

    ax.axis('square')
    ax.axis('off')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('iid action')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

    # iid noun
    ax = fig.add_subplot(2, 2, 2)

    num_class = len(np.unique(noun_iid))
    if num_class == 10:
        cmap = 'tab10'
    elif num_class == 2:
        cmap = colors.ListedColormap(['red', 'blue'])
    else:
        cmap = 'rainbow'

    for i in range(len(classes_noun)):
        cond = noun_iid == i
        ax.scatter(tx_iid[cond], ty_iid[cond], cmap=cmap, s=4.0, alpha=1.0, label=classes_noun[i])

    ax.axis('square')
    ax.axis('off')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('iid object')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

    # ood verb
    ax = fig.add_subplot(2, 2, 3)

    num_class = len(np.unique(verb_ood))
    if num_class == 10:
        cmap = 'tab10'
    elif num_class == 2:
        cmap = colors.ListedColormap(['red', 'blue'])
    else:
        cmap = 'rainbow'

    for i in range(len(classes_verb)):
        cond = verb_ood == i
        ax.scatter(tx_ood[cond], ty_ood[cond], cmap=cmap, s=4.0, alpha=1.0, label=classes_verb[i])

    ax.axis('square')
    ax.axis('off')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('ood action')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

    # ood noun
    ax = fig.add_subplot(2, 2, 4)

    num_class = len(np.unique(noun_ood))
    if num_class == 10:
        cmap = 'tab10'
    elif num_class == 2:
        cmap = colors.ListedColormap(['red', 'blue'])
    else:
        cmap = 'rainbow'

    for i in range(len(classes_noun)):
        cond = noun_ood == i
        ax.scatter(tx_ood[cond], ty_ood[cond], cmap=cmap, s=4.0, alpha=1.0, label=classes_noun[i])

    ax.axis('square')
    ax.axis('off')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('ood object')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

    # save
    figname = prefix+'.png'
    plt.savefig(figname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print('Save tsne to {}'.format(figname))

    # data = [tx_iid, ty_iid, tx_ood, ty_ood, noun_iid, verb_iid, feat_ood, noun_ood, classes_noun, classes_verb]
    # np.savez(prefix, *data)

    with open(f'{prefix}.pkl', 'wb') as f:
        pickle.dump([tx_iid, ty_iid, tx_ood, ty_ood, noun_iid, verb_iid, noun_ood, verb_ood, classes_noun, classes_verb], f)


def show_block(model, args, loader, verb_block, prefix, nsample=32):
    model.eval()

    # data
    num_blk = int(verb_block.max()+1)
    collect_first_img = [list() for _ in range(num_blk)]
    collect_second_img = [list() for _ in range(num_blk)]
    collect_first_mask = [list() for _ in range(num_blk)]
    collect_second_mask = [list() for _ in range(num_blk)]
    collect_count = [0 for _ in range(num_blk)]
    for batch in loader:
        if min(collect_count) >= nsample:
            break
        first_img, second_img, label, noun, first_mask, second_mask = batch
        blk = verb_block[label]
        for i in range(num_blk):
            if collect_count[i] >= nsample:
                continue
            mask_blk = blk == i
            cnt_blk = mask_blk.sum().item()
            add_blk = min(cnt_blk, nsample - collect_count[i])
            collect_first_img[i].append(first_img[mask_blk][:add_blk])
            collect_second_img[i].append(second_img[mask_blk][:add_blk])
            collect_first_mask[i].append(first_mask[mask_blk][:add_blk])
            collect_second_mask[i].append(second_mask[mask_blk][:add_blk])
            collect_count[i] += add_blk

    batch_first_img = torch.cat([torch.cat(x) for x in collect_first_img])
    batch_second_img = torch.cat([torch.cat(x) for x in collect_second_img])
    batch_first_mask = torch.cat([torch.cat(x) for x in collect_first_mask])
    batch_second_mask = torch.cat([torch.cat(x) for x in collect_second_mask])

    if torch.cuda.is_available():
        batch_first_img = batch_first_img.cuda()
        batch_second_img = batch_second_img.cuda()
        batch_first_mask = batch_first_mask.cuda()
        batch_second_mask = batch_second_mask.cuda()

    with torch.no_grad():
        if args.mask:
            _, _, p1, p2 = model(batch_first_img, batch_second_img, batch_first_mask, batch_second_mask)
        else:
            _, _, p1, p2 = model(batch_first_img, batch_second_img)

    # difference between feature pairs
    dp = (p1 - p2).abs()
    mu = dp.mean(dim=1, keepdim=True)
    sigma = dp.std(dim=1, keepdim=True)
    dp_standardized = (dp - mu) / sigma

    ax = sns.heatmap(dp_standardized.cpu().numpy(), cmap="YlGnBu", square=True, cbar=False)
    ax.hlines(np.cumsum(collect_count)[:-1], *ax.get_xlim(), colors='k', linestyles='dashed', linewidth=0.2)
    ax.set_xlabel('latent features')
    ax.set_ylabel('ordered instances')
    ax.axis('off')

    # save
    foldername = 'fig/latent/'
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    figname = foldername + prefix+'.png'
    plt.savefig(figname, bbox_inches='tight', pad_inches=0)
    print('Save tsne to {}'.format(figname))


def show_slot(img0, img1, rec0, rec1, mask0, mask1, figname='figslot.png'):
    cm = ColorMap(15, name='Dark2')
    fig, axes = plt.subplots(8, 6, figsize=(12, 16))
    fig.tight_layout(pad=0.0)
    m0 = cm(mask0.argmax(1)).detach().cpu()
    m1 = cm(mask1.argmax(1)).detach().cpu()
    im0 = img0.detach().cpu().permute(0, 2, 3, 1)
    im1 = img1.detach().cpu().permute(0, 2, 3, 1)

    for i, ax in enumerate(axes):
        ax[0].imshow(im0[i])
        ax[0].axis('off')
        ax[1].imshow(im1[i])
        ax[1].axis('off')
        ax[2].imshow(rec0[i].detach().cpu().permute(1, 2, 0))
        ax[2].axis('off')
        ax[3].imshow(rec1[i].detach().cpu().permute(1, 2, 0))
        ax[3].axis('off')
        ax[4].imshow(m0[i].permute(1, 2, 0))
        ax[4].axis('off')
        ax[5].imshow(m1[i].permute(1, 2, 0))
        ax[5].axis('off')

    plt.savefig(figname)


class ColorMap():

    def __init__(self, num_objects: int, name='hsv'):

        self.num_colors = num_objects
        self.cmap = plt.cm.get_cmap(name, self.num_colors + 1)
        self.r_get = {i: self.cmap(i)[0] for i in range(self.num_colors)}.get
        self.g_get = {i: self.cmap(i)[1] for i in range(self.num_colors)}.get
        self.b_get = {i: self.cmap(i)[2] for i in range(self.num_colors)}.get

    def __call__(self, inputs: Tensor) -> Tensor:
        """Function returns rgb image from uint8 segmentation input.

        Args:
            inputs (Tensor): Input segmenation. Shape: [batch_size, 1, height, width]

        Returns:
            Tensor: [batch_size, 3, height, width]
        """

        is_cuda = inputs.is_cuda

        inputs = inputs.repeat(1, 3, 1, 1).float()

        if is_cuda:
            inputs = inputs.to("cpu")

        inputs[:, 0, :, :].apply_(self.r_get)
        inputs[:, 1, :, :].apply_(self.g_get)
        inputs[:, 2, :, :].apply_(self.b_get)

        if is_cuda:
            inputs = inputs.to("cuda")

        return inputs
