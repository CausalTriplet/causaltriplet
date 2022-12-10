#!/usr/bin/env python3

import glob
import random
import logging
import collections
import pandas as pd
from skimage import io
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import torch

import pdb


def df_to_dict(df):
    dict_verb = collections.defaultdict(list)
    dict_noun = collections.defaultdict(list)
    dict_verb_noun = collections.defaultdict(list)
    # append
    for _, row in df.iterrows():
        dict_verb[row['verb_class']].append(row.name)
        dict_noun[row['noun_class']].append(row.name)
        dict_verb_noun[(row['verb_class'], row['noun_class'])].append(row.name)
    return dict_verb, dict_noun, dict_verb_noun


def split_df(df, axis='verb', seed=0):
    dict_verb, dict_noun, dict_verb_noun = df_to_dict(df)
    if axis == 'verb':
        keys = dict_verb
        ratio = 0.4
    elif axis == 'noun':
        keys = dict_noun
        ratio = 0.4
    elif axis == 'comp':
        keys = dict_verb_noun
        ratio = 0.3
    elif axis == 'loca':
        sz_img = 672

        top_left = (df['xmax'] * 2 < sz_img) & (df['ymax'] * 2 < sz_img)
        bot_leff = (df['xmax'] * 2 < sz_img) & (df['ymin'] * 2 > sz_img)
        top_right = (df['xmin'] * 2 > sz_img) & (df['ymax'] * 2 < sz_img)
        bot_right = (df['xmin'] * 2 > sz_img) & (df['ymin'] * 2 > sz_img)

        df['location_index'] = -1
        df.loc[top_left, 'location_index'] = 0
        df.loc[top_right, 'location_index'] = 1
        df.loc[bot_leff, 'location_index'] = 2
        df.loc[bot_right, 'location_index'] = 3

        # pdb.set_trace()
        # idx_ood = df['location_index'].isin([0,1])            # top vs bottom
        idx_ood = df['location_index'].isin([0, 2])              # left vs right
        # idx_ood = df['location_index'] == (seed % 4)
        # idx_ood = df.noun_index % 4 == ((df['location_index']) % 4)
        # idx_ood = df.verb_index % 4 == ((df['location_index']) % 4)
        idx_iid = (df['location_index'] >= 0) & (~idx_ood)
        return df.loc[idx_iid].copy(), df.loc[idx_ood].copy()
    else:
        # no ood split
        ratio = 0.1
        num_instance = len(df)
        idx_ood = random.sample(range(0, num_instance), max(10, int(ratio*num_instance)))   # remove ood split
        idx_iid = [i for i in range(num_instance) if i not in idx_ood]
        return df.iloc[idx_iid].copy(), df.iloc[idx_ood].copy()

    sort_keys = sorted(keys, key=lambda k: len(keys[k]), reverse=True)

    if axis == 'verb' or axis == 'noun':
        # split ood keys according to instance counts
        num_keys = len(sort_keys)
        num_ood = max(int(num_keys * ratio), 2)
        ood_keys = sort_keys[num_keys//2-num_ood//2:num_keys//2+num_ood//2]
        df_iid = df[~df[axis+'_class'].isin(ood_keys)]
        df_ood = df[df[axis+'_class'].isin(ood_keys)]
    elif axis == 'comp':
        from procthor.action import action_symmetry
        paired_dict = action_symmetry()

        random.shuffle(sort_keys)
        num_ood = int(len(sort_keys)*ratio)
        candidate_keys = sort_keys[:num_ood]
        iid_keys = sort_keys[num_ood:]

        iid_verb_set = {verb for verb, noun in iid_keys}
        iid_noun_set = {noun for verb, noun in iid_keys}
        iid_set = {(verb, noun) for verb, noun in iid_keys}

        ood_keys = list()
        for i, (verb, noun) in enumerate(candidate_keys):
            if (verb in paired_dict.keys()) and (verb in iid_verb_set) and (noun in iid_noun_set) and (paired_dict[verb], noun) in iid_set:
                ood_keys.append((verb, noun))
            else:
                iid_set.add((verb, noun))
                iid_verb_set.add(verb)
                iid_noun_set.add(noun)

        df['verb_noun_class'] = list(zip(df.verb_class, df.noun_class))
        df_iid = df[~df['verb_noun_class'].isin(ood_keys)]
        df_ood = df[df['verb_noun_class'].isin(ood_keys)]
    else:
        raise ValueError('ood axis not available')

    return df_iid, df_ood


def extract_pair(start_name, stop_name, transform=None, bbox=None):

    if transform:
        start_image = Image.open(start_name)
        if bbox:
            start_image = start_image.crop(bbox)
        start_image = transform(start_image)
    else:
        start_image = io.imread(start_name)

    if transform:
        stop_image = Image.open(stop_name)
        if bbox:
            stop_image = stop_image.crop(bbox)
        stop_image = transform(stop_image)
    else:
        stop_image = io.imread(stop_name)

    return start_image, stop_image


class ThresholdTransform(object):
    def __init__(self, thr):
        self.thr = thr / 255.           # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        return (x > self.thr).to(x.dtype)   # do not change the data type


class ActionDataset(Dataset):
    def __init__(self, dataset, df, foldername, transform, dict_noun_index, dict_verb_index, mask=False, bbox=False):
        """
            Dataset for action reasoning
        """
        self.dataset = dataset
        self.df = df.reset_index()
        self.foldername = foldername
        self.transform = transform
        self.noun_index = dict_noun_index
        self.verb_index = dict_verb_index
        assert not (mask and bbox)
        self.ismask = mask
        self.isbbox = bbox

    def _load_pairs(self, instance):
        if self.dataset == 'procthor':
            start_figname = os.path.join(self.foldername, instance.scene, 'color', instance.figure.split('second')[0] + 'first.png')
            stop_figname = os.path.join(self.foldername, instance.scene, 'color', instance.figure + '.png')
        elif self.dataset == 'epickitchens':
            start_figname = os.path.join(self.foldername, instance.participant_id, 'rgb_frames', instance.video_id, f'frame_{instance.start_frame:010d}.jpg')
            stop_figname = os.path.join(self.foldername, instance.participant_id, 'rgb_frames', instance.video_id, f'frame_{instance.stop_frame:010d}.jpg')
        else:
            raise NotImplementedError

        if self.isbbox:
            start_image, stop_image = extract_pair(start_figname, stop_figname, self.transform,
                                                   bbox=(instance.xmin, instance.ymin, instance.xmax, instance.ymax))
        else:
            start_image, stop_image = extract_pair(start_figname, stop_figname, self.transform)

        if self.ismask:
            start_maskname = os.path.join(self.foldername, instance.scene, 'mask', instance.figure.split('second')[0] + 'first.png')
            stop_maskname = os.path.join(self.foldername, instance.scene, 'mask', instance.figure + '.png')
            transform = transforms.Compose([
                transforms.Resize([7, 7]),          # resnet feature map
                transforms.ToTensor(),
                ThresholdTransform(thr=1)
            ])
            start_mask, stop_mask = extract_pair(start_maskname, stop_maskname, transform)
        else:
            start_mask, stop_mask = torch.zeros(1), torch.zeros(1)
        return start_image, stop_image, self.verb_index[instance.verb_class], self.noun_index[instance.noun_class], start_mask, stop_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        instance = self.df.loc[index]
        first_img, second_img, verb, noun, first_mask, second_mask = self._load_pairs(instance)
        return first_img, second_img, verb, noun, first_mask, second_mask


class PairDataset(Dataset):
    def __init__(self, df, foldername, transform_anchor, transform_sample, bbox=False):
        """
            Inputs:
        """
        self.df = df.reset_index()
        self.foldername = foldername
        self.transform_anchor = transform_anchor
        self.transform_sample = transform_sample
        self.isbbox = bbox

    def _load_contrastive(self, instance):
        start_name = os.path.join(self.foldername, instance.scene, 'color', instance.figure.split('second')[0] + 'first.png')
        stop_name = os.path.join(self.foldername, instance.scene, 'color', instance.figure + '.png')

        start_image = Image.open(start_name)
        if self.isbbox:
            start_image = start_image.crop((instance.xmin, instance.ymin, instance.xmax, instance.ymax))
        anchor_img = self.transform_anchor(start_image)
        # positive_img = self.transform_sample(start_image)
        positive_img = self.transform_anchor(start_image)

        stop_image = Image.open(stop_name)
        if self.isbbox:
            stop_image = stop_image.crop((instance.xmin, instance.ymin, instance.xmax, instance.ymax))
        # negative_img = self.transform_sample(stop_image)
        negative_img = self.transform_anchor(stop_image)

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        instance = self.df.loc[index]
        anchor_img, positive_img, negative_img = self._load_contrastive(instance)
        return anchor_img, positive_img, negative_img


class ImageDataset(Dataset):
    def __init__(self, df, foldername, transform, bbox=False):
        """
            Inputs:
        """
        self.df = df.reset_index()
        self.foldername = foldername
        self.transform = transform
        self.isbbox = bbox

    def _load_image(self, figname, instance):
        img = Image.open(figname)
        if self.isbbox:
            img = img.crop((instance.xmin, instance.ymin, instance.xmax, instance.ymax))
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.df) * 2

    def __getitem__(self, index):
        instance = self.df.loc[index//2]
        if index % 2 == 0:
            figname = os.path.join(self.foldername, instance.scene, 'color', instance.figure.split('second')[0] + 'first.png')
        else:
            figname = os.path.join(self.foldername, instance.scene, 'color', instance.figure + '.png')
        img = self._load_image(figname, instance)
        return img, instance.noun_index


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def dict_to_stat(dict_data):
    dict_stat = dict()
    for key, value in dict_data.items():
        dict_stat[key] = len(value)
    dict_stat = dict(sorted(dict_stat.items(), key=lambda item: item[1], reverse=True))
    return dict_stat


def balance_stat(dict_data, stat_data):
    num_total = sum(stat_data.values())
    num_class = len(stat_data.keys())
    num_max = int(num_total / num_class / 1.5)
    for key, value in dict_data.items():
        if len(value) > num_max:
            dict_data[key] = random.sample(value, num_max)
    return dict_data


def load_categories(foldername):

    verb_df_path = f"{foldername}/EPIC_100_verb_classes.csv"
    noun_df_path = f"{foldername}/EPIC_100_noun_classes.csv"

    verb_df = pd.read_csv(verb_df_path)
    noun_df = pd.read_csv(noun_df_path)

    return verb_df, noun_df


def load_annotations(foldername):

    train_100_path = f"{foldername}/EPIC_100_train.pkl"
    valid_100_path = f"{foldername}/EPIC_100_validation.pkl"

    train_100_df: pd.DataFrame = pd.read_pickle(train_100_path)
    valid_100_df: pd.DataFrame = pd.read_pickle(valid_100_path)

    groundtruth_df = pd.concat([train_100_df, valid_100_df], axis=0)

    if "narration_id" in groundtruth_df.columns:
        groundtruth_df.set_index("narration_id", inplace=True)

    # df = groundtruth_df.sort_values(by=['narration_timestamp'])
    df = groundtruth_df.sort_values(by=['narration_id'])

    return df


def load_preprocess(foldername, thres):

    # pattern = foldername / '*.pkl'
    pattern = f'{foldername}/*.hdf'
    files = glob.glob(str(pattern))

    stack_df = list()
    for file in files:
        # df_raw: pd.DataFrame = pd.read_pickle(file)
        df_raw: pd.DataFrame = pd.read_hdf(file)
        df_sub = df_raw[(df_raw['start_score'] > thres) & (df_raw['stop_score'] > thres)]
        stack_df.append(df_sub)
        # pdb.set_trace()
        # save_attributes(df_raw, 'prepro/'+file.split('/')[1][:-3]+'hdf')

    df = pd.concat(stack_df, axis=0)

    return df


def verb_set():
    verbs = {"open", "close", "cut", "move", "remove", "adjust",
             "peel", "empty", "flip", "fill", "fold", "pull", "break", "lift", "wrap"
             "hang", "add", "roll", "stretch", "divide", "sharpen", "attach", "increase", "bend", "flatten"}
    return verbs
