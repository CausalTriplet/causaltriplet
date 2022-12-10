import os
import numpy as np
from tqdm import tqdm
from ai2thor.controller import Controller
import random
import cv2
import prior
import pickle

import torch
import torch.optim as optim
from torch.nn import functional
from torchvision import transforms

import sys
import csv
import argparse
import time
import glob
import pandas as pd

import logging

from visualize import *
from dataset import *
from util import *

from model.respair import ResPair
from model.clippair import ClipPair, convert_models_to_fp32
from model.adversary import Adversary
from sparsity.block import sparse_criterion_with_label

import pdb


def parse_arguments():
    parser = argparse.ArgumentParser('Parse main configuration file', add_help=False)
    # setting
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--ood", default='noun', type=str)
    parser.add_argument("--dataset", default='procthor', type=str)
    parser.add_argument("--translation", default=0.0, type=float)
    parser.add_argument("--path_data", default='./data', type=str)
    parser.add_argument("--train_size", default=5000, type=int, help='size of training data')
    # model
    parser.add_argument("--model", default=None, type=str, help='resnet18, resnet50, clip, groupaverage')
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--linear", default=False, action='store_true')
    parser.add_argument("--mask", default=False, action='store_true')
    parser.add_argument("--bbox", default=False, action='store_true')
    # loader
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    # train
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--finetune", default=False, action='store_true')
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--lr", default=0.0002, type=float)
    parser.add_argument("--critic_action", default=0.0, type=float)
    parser.add_argument("--critic_state", default=0.0, type=float)
    parser.add_argument("--sparse", default=0.0, type=float)
    parser.add_argument("--amin", default=0.0, type=float, help='min value for attention map')
    # log
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    return parser.parse_args()


def set_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

    if args.ckpt:
        ckpt = args.ckpt.split('/')[-1][:-3].replace('_', '-')
    else:
        ckpt = 'public'

    logname = f'{args.ood}_{args.train_size}_{args.model}_{args.dim}_tran_{args.translation}_critic_{args.critic_action}_{args.critic_state}_linear_{args.linear}_mask_{args.mask}_bbox_{args.bbox}_encoder_{args.finetune}_pretrain_{ckpt}_sparse_{args.sparse}_amin_{args.amin}_seed_{args.seed}'

    print(f'Save log into {logname}')

    if args.dataset == 'procthor':
        handler = logging.FileHandler(f'log/thor/{logname}.log', 'w', 'utf-8')
    elif args.dataset == 'epickitchens':
        handler = logging.FileHandler(f'log/epic/{logname}.log', 'w', 'utf-8')
    else:
        raise NotImplementedError
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.debug(f'{time.asctime(time.localtime())}')

    logging.info(args)


def set_loader(args, ratio_valid=0.2):

    # load meta data
    if args.dataset == 'procthor':
        '''
            procthor: ['scene', 'idx', 'figure', 'noun_class', 'verb_class', 'xmin', 'ymin', 'xmax', 'ymax']
        '''
        filecsv = f'{args.path_data}/annotations.csv'
        if os.path.exists(filecsv):
            df = pd.read_csv(filecsv)
            logging.info(f'Loaded annotations from {filecsv}')
        else:
            files = glob.glob(f'{args.path_data}/proc_*/annotations.csv')
            files.sort()
            stack = list()
            for file in files:
                df = pd.read_csv(file, header=None, names=['scene', 'idx', 'figure', 'noun_class', 'verb_class', 'xmin', 'ymin', 'xmax', 'ymax'])
                stack.append(df)
            df = pd.concat(stack, ignore_index=True)
            df.to_csv(filecsv, index=False)

        df = df[df['verb_class'] != 'none']
        df = df[df['verb_class'] != 'cook']

    elif args.dataset == 'epickitchens':
        '''
            epickitchens: ['participant_id', 'video_id', 'narration_timestamp', 'start_timestamp',
                           'stop_timestamp', 'start_frame', 'stop_frame', 'narration', 'verb',
                           'verb_class', 'noun', 'noun_class', 'all_nouns', 'all_noun_classes',
                           'start_score', 'stop_score']
        '''
        df_verb, df_noun = load_categories('epickitchens/annotations')
        df = load_preprocess('epickitchens/prepro', thres=0.5)
        # align var name
        df.rename(columns={'noun_class': 'noun_index', 'verb_class': 'verb_index'}, inplace=True)
        df['noun_class'] = df.apply(lambda row: df_noun.loc[row.noun_index].key, axis=1)
        df['verb_class'] = df.apply(lambda row: df_verb.loc[row.verb_index].key, axis=1)

        # print(df)
        # df.to_csv('epicframes.csv')
        # pdb.set_trace()
    else:
        raise NotImplementedError

    # attributes
    dict_noun_index = {k: v for v, k in enumerate(df['noun_class'].unique())}
    dict_noun_class = {v: k for v, k in enumerate(df['noun_class'].unique())}
    dict_verb_index = {k: v for v, k in enumerate(df['verb_class'].unique())}
    dict_verb_class = {v: k for v, k in enumerate(df['verb_class'].unique())}
    df['noun_index'] = df.apply(lambda row: dict_noun_index[row.noun_class], axis=1)
    df['verb_index'] = df.apply(lambda row: dict_verb_index[row.verb_class], axis=1)

    num_instance = len(df)
    num_noun = len(df['noun_class'].unique())
    num_verb = len(df['verb_class'].unique())
    logging.info(f'Dataset stat: # instance {num_instance}, # noun {num_noun}, # verb {num_verb}')

    # symmetry
    if args.dataset == 'procthor':
        # pdb.set_trace()
        from procthor.action import action_symmetry
        symmetric_verb_class = action_symmetry()
        symmetric_verb_index = {dict_verb_index[k]: dict_verb_index[v] for k, v in symmetric_verb_class.items()}
    else:
        symmetric_verb_index = None

    # rebalance data
    dict_verb, dict_noun, dict_verb_noun = df_to_dict(df)
    stat_verb = dict_to_stat(dict_verb)
    show_stat(stat_verb, figname=f'fig/stat_{args.dataset}_verb')

    # feasible combinations
    bool_verb_noun = torch.zeros((num_verb, num_noun)).bool()
    for (verb, noun) in dict_verb_noun.keys():
        bool_verb_noun[(dict_verb_index[verb], dict_noun_index[noun])] = True

    dict_verb = balance_stat(dict_verb, stat_verb)
    stat_verb = dict_to_stat(dict_verb)
    show_stat(stat_verb, figname=f'fig/stat_{args.dataset}_verb_reb')

    indices = [name for names in dict_verb.values() for name in names]
    logging.info(f'{len(indices)} / {len(df)} instances are kept from rebalance')
    df = df[df.index.isin(indices)].reset_index(drop=True)

    # block preprocessing
    num_blk = num_verb - int(len(symmetric_verb_index) / 2)
    verb_block = -torch.ones(num_verb, dtype=torch.uint8)
    cnt = 0
    for idx_verb in range(num_verb):
        if idx_verb in symmetric_verb_index and symmetric_verb_index[idx_verb] < idx_verb:
            verb_block[idx_verb] = verb_block[symmetric_verb_index[idx_verb]]
        else:
            verb_block[idx_verb] = cnt
            cnt += 1
    assert verb_block.max().int() + 1 == num_blk
    logging.info(f'{num_blk} blocks of latent variables')

    # pdb.set_trace()

    # ood instances
    df_iid, df_ood = split_df(df, axis=args.ood, seed=args.seed)
    dict_verb_iid, dict_noun_iid, dict_verb_noun_iid = df_to_dict(df_iid)
    dict_verb_ood, dict_noun_ood, dict_verb_noun_ood = df_to_dict(df_ood)

    # show_split(dict_verb_noun_iid, dict_verb_noun_ood, figname=f'fig/split/test.png')
    # show_split(dict_verb_noun_iid, dict_verb_noun_ood, figname=f'fig/split/split_{args.dataset}_{args.ood}_{args.seed}.pdf')

    # split
    max_num_ood = min(5000, int(0.5 * len(df_iid)))
    num_valid = max(min(len(df_ood), max_num_ood), 1)

    df_iid = df_iid.sample(frac=1)  # shuffle order
    df_train = df_iid[num_valid:]
    df_test = df_iid[:num_valid]

    # ood validation set for model selection
    if len(df_ood) < num_valid * 2:
        logging.warning("duplication between ood validation and ood test")
    else:
        logging.info("disjoint ood validation and test set")
    df_valid = df_ood[-num_valid:]
    df_ood = df_ood[:num_valid]

    df_train = df_train[:args.train_size]

    # transform
    if args.model is None:
        transform = None
    elif args.model[:3] in ['res', 'vit'] or args.model[:5] == 'group':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean=mean, std=std)
        if args.translation > 0.0 and not args.bbox:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(1.-args.translation, 1.)),
                transforms.ToTensor(),
                normalize,
            ])
            # todo: clean up logic
        else:
            transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ])
    elif args.model[:4] == 'slot':
        transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
        ])
        # transforms.ConvertImageDtype(dtype=torch.float32)
    elif args.model[:4] == 'clip':
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = None

    data_train = ActionDataset(args.dataset, df_train, args.path_data, transform, dict_noun_index, dict_verb_index, args.mask, args.bbox)
    data_test = ActionDataset(args.dataset, df_test, args.path_data, transform, dict_noun_index, dict_verb_index, args.mask, args.bbox)
    data_ood = ActionDataset(args.dataset, df_ood, args.path_data, transform, dict_noun_index, dict_verb_index, args.mask, args.bbox)
    data_valid = ActionDataset(args.dataset, df_valid, args.path_data, transform, dict_noun_index, dict_verb_index, args.mask, args.bbox)

    # # data sanity check
    # data_train[0]
    # pdb.set_trace()

    logging.info(f'# train: {len(data_train)}      # test: {len(data_test)}        # ood: {len(data_ood)}')

    # create loader
    loader_train = torch.utils.data.DataLoader(
        data_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker)
    loader_test = torch.utils.data.DataLoader(
        data_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker)
    loader_ood = torch.utils.data.DataLoader(
        data_ood, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker)
    loader_valid = torch.utils.data.DataLoader(
        data_valid, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker)

    return loader_train, loader_test, loader_ood, loader_valid, dict_noun_class, dict_verb_class, symmetric_verb_index, verb_block, bool_verb_noun


def set_model(args, num_action, num_object):
    if args.model[:3] in ['res', 'vit']:
        model = ResPair(feat_dim=args.dim, num_class=num_action, name=args.model, linear=args.linear, amin=args.amin)
    elif args.model[:4] == 'clip':
        model = ClipPair(feat_dim=args.dim, num_class=num_action, linear=args.linear)
    elif args.model[:4] == 'slot':
        from model.slotpair import SlotAutoEncoder, SlotAverage, SlotMatchMean, SlotMatchMax
        if args.model == 'slotaverage':
            model = SlotAverage(device='cuda', resolution=(128, 128), num_class=num_action).cuda()
        elif args.model == 'slotmatchmean':
            model = SlotMatchMean(device='cuda', resolution=(128, 128), num_class=num_action).cuda()
        elif args.model == 'slotmatchmax':
            model = SlotMatchMax(device='cuda', resolution=(128, 128), num_class=num_action).cuda()
        else:
            raise NotImplementedError
    elif args.model[:5] == 'group':
        # config
        import pickle
        file = open("groupvit/cfg.pkl", 'rb')
        cfg = pickle.load(file)
        file.close()

        # dependency
        import os.path as osp
        cwd = os.path.dirname(osp.abspath(__file__))
        gvitdir = osp.join(cwd, 'groupvit')
        sys.path.insert(0, gvitdir)

        from mmcv.cnn.utils import revert_sync_batchnorm
        from models import build_model
        from utils import load_checkpoint

        # encoder
        groupvit = build_model(cfg.model)
        groupvit = revert_sync_batchnorm(groupvit)
        load_checkpoint(cfg, groupvit, None, None)

        # model
        from model.grouppair import GroupAverage, GroupDense, GroupMatchTokenMax, GroupMatchTokenMean
        if args.model == 'groupaverage':
            model = GroupAverage(groupvit.img_encoder, feat_dim=args.dim, num_class=num_action)
        elif args.model == 'groupdense':
            model = GroupDense(groupvit.img_encoder, feat_dim=args.dim, num_class=num_action)
        elif args.model == 'grouptokenmax':
            model = GroupMatchTokenMax(groupvit.img_encoder, feat_dim=args.dim, num_class=num_action)
        elif args.model == 'grouptokenmean':
            model = GroupMatchTokenMean(groupvit.img_encoder, feat_dim=args.dim, num_class=num_action)
        else:
            raise NotImplementedError

    else:
        model = None

    critic_action = Adversary(feat_dim=args.dim, num_class=num_object)
    critic_state = Adversary(feat_dim=args.dim, num_class=num_object)

    if torch.cuda.is_available():
        model = model.cuda()
        critic_action = critic_action.cuda()
        critic_state = critic_state.cuda()
        if args.model[:4] == 'clip':
            convert_models_to_fp32(model.encoder)

    if args.ckpt:
        if args.model[:4] == 'slot':
            ckpt_dict = torch.load(args.ckpt)['model']
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            ckpt_dict = {k: v for k, v in ckpt_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(ckpt_dict)
            # 3. load the new state dict
            info = model.load_state_dict(model_dict)

            # checkpoint = torch.load(args.ckpt)
            # info = model.load_state_dict(checkpoint['model'])
            logging.info(f'Loaded encoder from {args.ckpt}: {info}')
        else:
            state_dict = torch.load(args.ckpt)
            if args.epochs > 0:
                # encoder
                encoder_dict = {}
                for k, v in state_dict.items():
                    if k[:7] == "encoder":
                        k = k.replace("encoder.", "")
                        encoder_dict[k] = v
                info = model.encoder.load_state_dict(encoder_dict, strict=True)
                logging.info(f'Loaded encoder from {args.ckpt}: {info}')
                # proj
                ckptname = args.ckpt.split('/')[-1][:-3]
                if ckptname[:8] == 'contrast':
                    proj_dict = {}
                    for k, v in state_dict.items():
                        if k[:4] == "proj":
                            k = k.replace("proj.", "")
                            proj_dict[k] = v
                    info = model.proj.load_state_dict(proj_dict, strict=True)
                    logging.info(f'Loaded proj head from {args.ckpt}: {info}')
            else:
                # model
                info = model.load_state_dict(state_dict, strict=True)
                logging.warning(f'Loaded model from {args.ckpt}: {info}')

    return model, critic_action, critic_state


def train(model, critic_action, critic_state, optim_model, optim_critic_action, optim_critic_state, criterion, loader, args, verb_block, bool_verb_noun):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':4.2f')
    acc_action = AverageMeter('Act', ':4.2f')
    acc_object = AverageMeter('Obj', ':4.2f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, acc_action, acc_object])

    uniform_verb_noun = bool_verb_noun.float()
    uniform_verb_noun /= uniform_verb_noun.sum(axis=1, keepdim=True)

    model.train()

    loss_sum = 0.0
    acc_sum_act = 0.0
    acc_sum_obj_in_state = 0.0
    acc_sum_obj_in_action = 0.0
    loss_sum_sparse = 0.0

    end = time.time()
    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # data
        first_img, second_img, label, noun, first_mask, second_mask = batch
        if torch.cuda.is_available():
            first_img = first_img.cuda()
            second_img = second_img.cuda()
            label = label.cuda()
            noun = noun.cuda()
            first_mask = first_mask.cuda()
            second_mask = second_mask.cuda()

        if args.ood == 'comp':

            if args.mask:
                _, feat, p1, p2 = model(first_img, second_img, first_mask, second_mask)
            else:
                _, feat, p1, p2 = model(first_img, second_img)

            # --- action critic ---

            # object classifier
            output_critic_action = critic_action(feat.detach())
            loss_critic_action = criterion(output_critic_action, noun)
            optim_critic_action.zero_grad()
            loss_critic_action.backward()
            optim_critic_action.step()
            del loss_critic_action             # release memory

            # object metric
            _, predict = output_critic_action.max(1)
            correct = (predict == noun.to(torch.uint8)).float().sum()
            accuracy = correct / predict.size()[0]
            acc_sum_obj_in_action += accuracy.item()

            # --- state critic ---

            # object classifier
            output_critic_state = critic_state(torch.cat([p1.detach(), p2.detach()]))
            loss_critic_state = criterion(output_critic_state, torch.cat([noun, noun]))
            optim_critic_state.zero_grad()
            loss_critic_state.backward()
            optim_critic_state.step()
            del loss_critic_state             # release memory

            # object metric
            _, predict = output_critic_state.max(1)
            correct = (predict == torch.cat([noun, noun]).to(torch.uint8)).float().sum()
            accuracy = correct / predict.size()[0]
            acc_sum_obj_in_state += accuracy.item()

            # release memory
            del feat, p1, p2, output_critic_action, output_critic_state
            torch.cuda.empty_cache()

        # --------------------------------------

        # pdb.set_trace()

        # slots0, z0 = model(first_img)
        # slots1, z1 = model(second_img, slot_init=slots0)

        # recon_c0, recons0, mask0, slots0, z0 = model(first_img)
        # recon_c1, recons1, mask1, slots1, z1 = model(second_img, slot_init=slots0)
        # show_slot(first_img, second_img, recon_c0, recon_c1, mask0, mask1, figname='figs/figslot.png')

        # pdb.set_trace()

        # action classifier
        if args.mask:
            opt_model, feat, p1, p2 = model(first_img, second_img, first_mask, second_mask)
        else:
            opt_model, feat, p1, p2 = model(first_img, second_img)

        # pdb.set_trace()

        loss_model = criterion(opt_model, label)
        loss = loss_model

        # TODO: refactor logic -> only if ood == comp
        # adversarial critic
        if args.critic_action > 0:
            # uniform distribution over all object classes
            # target = torch.ones_like(opt_critic).div(opt_critic.size(1))

            # adjusted uniform distribution over the feasible object classes
            # target = uniform_verb_noun[label]
            # if torch.cuda.is_available():
            #     target = target.cuda()

            output_critic_action = critic_action(feat)
            loss_critic_action = criterion(output_critic_action, noun)
            loss -= loss_critic_action * args.critic_action

        if args.critic_state > 0:
            output_critic_state = critic_state(torch.cat([p1, p2]))
            loss_critic_state = criterion(output_critic_state, torch.cat([noun, noun]))
            loss -= loss_critic_state * args.critic_state

        # block sparsity
        if args.sparse > 0:
            blk = functional.one_hot(verb_block[label].to(torch.int64), num_classes=verb_block.max().int()+1)
            if torch.cuda.is_available():
                blk = blk.cuda()
            loss_sparsity = sparse_criterion_with_label(p1, p2, blk) * args.sparse
            loss += loss_sparsity
        else:
            loss_sparsity = torch.tensor(0.0)

        optim_model.zero_grad()
        loss.backward()
        optim_model.step()

        # action metric
        loss_sum += loss_model.item()
        _, predict = opt_model.max(1)
        correct = (predict == label.to(torch.uint8)).float().sum()
        accuracy = correct / label.size()[0]
        acc_sum_act += accuracy.item()
        acc_action.update(accuracy, label.size(0))

        loss_sum_sparse += loss_sparsity.item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i > 0 and i % args.print_freq == 0:
            progress.display(i+1)

    return acc_sum_act / len(loader), loss_sum / len(loader), acc_sum_obj_in_action / len(loader), acc_sum_obj_in_state / len(loader), loss_sum_sparse / len(loader)


def evaluate(model, critic_action, critic_state, criterion, loader, args, symmetric_index=None, reverse=False):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':4.2f')
    top1 = AverageMeter('Acc@1', ':4.2f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, top1])

    model.eval()

    loss_sum = 0.0
    acc_sum_act = 0.0
    acc_sum_obj = 0.0

    sym_sum = 0.0
    miss_sum = 0.0

    end = time.time()
    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # data
        first_img, second_img, label, noun, first_mask, second_mask = batch

        label_sym = label.clone()
        if symmetric_index is not None:
            label_sym.apply_(lambda x: symmetric_index[x] if x in symmetric_index else x)

        if reverse:
            assert symmetric_index
            flip = torch.zeros_like(label).gt(0)
            flip.apply_(lambda x: x in symmetric_index)
            label.apply_(lambda x: symmetric_index[x] if x in symmetric_index else x)
            first_img[flip], second_img[flip] = second_img[flip], first_img[flip]
            first_mask[flip], second_mask[flip] = second_mask[flip], first_mask[flip]

        if torch.cuda.is_available():
            first_img = first_img.cuda()
            second_img = second_img.cuda()
            label = label.cuda()
            label_sym = label_sym.cuda()
            noun = noun.cuda()
            first_mask = first_mask.cuda()
            second_mask = second_mask.cuda()

        # test
        with torch.no_grad():
            if args.mask:
                output, _, _, _ = model(first_img, second_img, first_mask, second_mask)
            else:
                output, _, _, _ = model(first_img, second_img)
        loss = criterion(output, label)

        # metric
        loss_sum += loss.item()
        _, predict = output.max(1)
        correct = (predict == label.to(torch.uint8)).float().sum()
        accuracy = correct / label.size(0)
        acc_sum_act += accuracy.item()
        top1.update(accuracy, label.size(0))

        # symmetric
        if symmetric_index is not None:
            mask = predict != label
            correct_sym = (predict[mask] == label_sym[mask].to(torch.uint8)).float().sum()
            miss_sum += mask.sum()
            sym_sum += correct_sym.item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i > 0 and i % args.print_freq == 0:
            progress.display(i+1)

    return acc_sum_act / len(loader), loss_sum / len(loader), acc_sum_obj / len(loader), sym_sum / max(1e-6, miss_sum)


def embedding(model, loader):

    stack_verb, stack_noun, stack_feat, stack_emb1, stack_emb2 = list(), list(), list(), list(), list()

    for i, batch in enumerate(loader):

        # data
        first_img, second_img, verb, noun, first_mask, second_mask = batch
        if torch.cuda.is_available():
            first_img = first_img.cuda()
            second_img = second_img.cuda()
            first_mask = first_mask.cuda()
            second_mask = second_mask.cuda()

        # feat
        with torch.no_grad():
            if args.mask:
                _, feat, emb1, emb2 = model(first_img, second_img, first_mask, second_mask)
            else:
                _, feat, emb1, emb2 = model(first_img, second_img)

        # stack
        stack_verb.append(verb)
        stack_noun.append(noun)
        stack_feat.append(feat.detach())
        stack_emb1.append(emb1.detach())
        stack_emb2.append(emb2.detach())

    stack_verb = torch.cat(stack_verb)
    stack_noun = torch.cat(stack_noun)
    stack_feat = torch.cat(stack_feat).cpu()
    stack_emb1 = torch.cat(stack_emb1).cpu()
    stack_emb2 = torch.cat(stack_emb2).cpu()

    return stack_verb, stack_noun, stack_feat, stack_emb1, stack_emb2


def tsne(model, loader_test, loader_ood, dict_noun_class, dict_verb_class, args, acc_test, acc_ood):

    model.eval()
    foldername = 'fig/tsne'
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    test_verb, test_noun, test_feat, test_emb1, test_emb2 = embedding(model, loader_test)
    ood_verb, ood_noun, ood_feat, ood_emb1, ood_emb2 = embedding(model, loader_ood)

    if args.ckpt:
        ckpt = args.ckpt.split('/')[-1][:-3].replace('_', '-')
    else:
        ckpt = 'public'

    figname = f'{args.ood}_{args.train_size}_{args.model}_{args.dim}_tran_{args.translation}_critic_{args.critic_action}_{args.critic_state}_linear_{args.linear}_mask_{args.mask}_bbox_{args.bbox}_encoder_{args.finetune}_pretrain_{ckpt}_sparse_{args.sparse}_amin_{args.amin}_seed_{args.seed}'

    figname += f'_iid_{acc_test:.2f}_ood_{acc_ood:.2f}'
    show_tsne(test_feat, test_noun, test_verb, ood_feat, ood_noun, ood_verb, dict_noun_class, dict_verb_class, foldername + f'/{figname}_feat')
    show_tsne(test_emb1, test_noun, test_verb, ood_emb1, ood_noun, ood_verb, dict_noun_class, dict_verb_class, foldername + f'/{figname}_emb1')
    show_tsne(test_emb2, test_noun, test_verb, ood_emb2, ood_noun, ood_verb, dict_noun_class, dict_verb_class, foldername + f'/{figname}_emb2')


def main(args):
    set_seed(args.seed)
    set_logger(args)
    loader_train, loader_test, loader_ood, loader_valid, dict_noun_class, dict_verb_class, symmetric_verb_index, verb_block, bool_verb_noun = set_loader(args)

    if args.model is None:
        output_path = f'fig/pairs/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for idx, (first, second, verb, noun, _, _) in enumerate(loader_test.dataset):
            show_pair(first, second, dict_verb_class[verb], dict_noun_class[noun], suffix=f'_iid_{idx}', savedir=output_path)
            if idx % 10 == 0:
                print(f'# {idx} / {len(loader_train.dataset)}')
            if idx >= 10:
                break
        for idx, (first, second, verb, noun, _, _) in enumerate(loader_ood.dataset):
            show_pair(first, second, dict_verb_class[verb], dict_noun_class[noun], suffix=f'_ood_{idx}', savedir=output_path)
            if idx % 10 == 0:
                print(f'# {idx} / {len(loader_train.dataset)}')
            if idx >= 10:
                break
    else:
        model, critic_action, critic_state = set_model(args, len(dict_verb_class), len(dict_noun_class))

        optim_model = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optim_critic_action = optim.Adam(critic_action.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optim_critic_state = optim.Adam(critic_state.parameters(), lr=args.lr, betas=(0.5, 0.999))
        criterion = torch.nn.CrossEntropyLoss()

        if args.finetune:
            if args.model[:4] == 'clip':
                grad_param(model.encoder.visual)
            else:
                grad_param(model)
        param_all, param_tra = count_param(model)
        logging.info(f'count parameters: total = {param_all}, trainable = {param_tra}')

        # acc_test, loss_test, _, acc_test_sym = evaluate(model, critic_action, critic_state, criterion, loader_test, args, symmetric_verb_index)
        # acc_ood, loss_ood, _, acc_ood_sym = evaluate(model, critic_action, critic_state, criterion, loader_ood, args, symmetric_verb_index)

        # acc_test, loss_test, _, acc_test_sym = evaluate(model, critic_action, critic_state, criterion, loader_test, args, symmetric_verb_index)
        # acc_ood, loss_ood, _, acc_ood_sym = evaluate(model, critic_action, critic_state, criterion, loader_ood, args, symmetric_verb_index)
        # logging.info(f'initialized model:  test = {acc_test:.2f}  ood = {acc_ood:.2f}  test_sym = {acc_test_sym:.2f}  ood_sym = {acc_ood_sym:.2f}')

        acc_test_best = 0.0

        end = time.time()
        for epoch in range(args.epochs):

            acc_train, loss_train, obj_train_action, obj_train_state, loss_sparsity = train(model, critic_action, critic_state, optim_model, optim_critic_action, optim_critic_state, criterion, loader_train, args, verb_block, bool_verb_noun)
            acc_test, loss_test, _, _ = evaluate(model, critic_action, critic_state, criterion, loader_test, args)
            acc_ood, loss_ood, _, _ = evaluate(model, critic_action, critic_state, criterion, loader_ood, args)

            # skip validation when the test set is too small or valid results always similar to test results
            # acc_valid, loss_valid, _, _ = evaluate(model, critic_action, critic_state, criterion, loader_valid, args)

            logging.info(f'epoch #{epoch:02d}  train = {acc_train:.4f}  test = {acc_test:.4f}  ood = {acc_ood:.4f}  val = {acc_ood:.4f}  obj = ({obj_train_action:.4f}, {obj_train_state:.4f})  sparsity = {loss_sparsity:.4f}  elapsed = {(time.time() - end):.0f}s')

            end = time.time()

            # save checkpoints
            if args.ood == 'full' and acc_test > acc_test_best:
                # if acc_test > acc_test_best:
                ckptname = f'{args.ood}_{args.train_size}_{args.model}_{args.dim}_tran_{args.translation}_critic_{args.critic_action}_{args.critic_state}_linear_{args.linear}_mask_{args.mask}_bbox_{args.bbox}_encoder_{args.finetune}_pretrain_{args.ckpt}_sparse_{args.sparse}_amin_{args.amin}_seed_{args.seed}'
                torch.save(model.state_dict(), f'ckpt/{ckptname}.pt')
                acc_test_best = acc_test

            # if epoch % 5 == 0:
            #     show_block(model, args, loader_train, verb_block, prefix=f'{args.ood}_{args.sparse}_train_{epoch}')
            #     show_block(model, args, loader_test, verb_block, prefix=f'{args.ood}_{args.sparse}_iid_{epoch}')
            #     show_block(model, args, loader_ood, verb_block, prefix=f'{args.ood}_{args.sparse}_ood_{epoch}')

        # tsne(model, loader_test, loader_ood, dict_noun_class, dict_verb_class, args, acc_test, acc_ood)

    print('Well done')


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    main(args)
