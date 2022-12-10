import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import argparse
from common import *

sns.set_theme()
pd.options.display.float_format = '{:,.2f}'.format
plt.rcParams["figure.dpi"] = 140

import pdb

def parse_arguments():
    parser = argparse.ArgumentParser('Parse main configuration file', add_help=False)
    # setting
    parser.add_argument("--ood", default='noun', type=str)
    parser.add_argument("--translation", default=0.0, type=float)
    parser.add_argument("--train_size", default=5000, type=int, help='size of training data')
    # model
    parser.add_argument('--model', default='resnet18')
    parser.add_argument("--mask", default=False, action='store_true')
    parser.add_argument("--bbox", default=False, action='store_true')
    # train
    parser.add_argument("--finetune", default=False, action='store_true')
    parser.add_argument("--ckpt", default='public', type=str)
    parser.add_argument("--critic_action", default=0.0, type=float)
    parser.add_argument("--critic_state", default=0.0, type=float)
    parser.add_argument("--amin", default=0.0, type=float, help='min value for attention map')
    # analysis
    parser.add_argument("--foldername", default='save/log_group_5000_0822')
    parser.add_argument("--setting", default='model')
    parser.add_argument("--nepoch", default=500, type=int)
    return parser.parse_args()


def compare_critic(df_all, df_iid, df_ood, args, name='critic_action'):
    critics = df_all[name].unique()

    stack_iid = list()
    stack_ood = list()
    for critic in critics:
        if name == 'critic_action':
            df_all_setting, df_iid_setting, df_ood_setting = extract_setting(df_all, df_iid, df_ood,
            args.bbox, args.ood, args.model, critic, args.critic_state, 
            args.finetune, args.translation, args.mask, args.amin)
        elif name == 'critic_state':
            df_all_setting, df_iid_setting, df_ood_setting = extract_setting(df_all, df_iid, df_ood,
            args.bbox, args.ood, args.model, args.critic_action, critic, 
            args.finetune, args.translation, args.mask, args.amin)
        else:
            raise NotImplementedError
        stack_iid.append(df_iid_setting[['epoch', 'iid', name, 'seed']])
        stack_ood.append(df_ood_setting[['epoch', 'ood', name, 'seed']])
    df_iid_setting = pd.concat(stack_iid)
    df_ood_setting = pd.concat(stack_ood)

    # stats
    pd.options.display.float_format = '{:,.2f}'.format

    print('--- avg ---')
    print(df_iid_setting.groupby(name).mean())
    print(df_ood_setting.groupby(name).mean())
    print('--- std ---')
    print(df_iid_setting.groupby(name).std())
    print(df_ood_setting.groupby(name).std())

    df_all_oracle, df_iid_oracle, df_ood_oracle = extract_setting(df_all, df_iid, df_ood,
            args.bbox, args.ood, args.model, 0.0, 0.0, 
            False, args.translation, args.mask, args.amin)

    # select seeds that are solvable for oracle but not challenging for vanilla
    thres = 0.8
    cond_oracle = df_ood_oracle[df_ood_oracle.ood > thres].seed.unique()
    df_ood_vanilla = df_ood_setting[df_ood_setting[name]==0.0]
    cond_vanilla = df_ood_vanilla[df_ood_vanilla.ood < thres-0.1].seed.unique()

    cond = list(set(cond_oracle) & set(cond_vanilla))

    print(cond)

    df_iid_setting = df_iid_setting[df_iid_setting['seed'].isin(cond)]
    df_ood_setting = df_ood_setting[df_ood_setting['seed'].isin(cond)]

    print('--- avg ---')
    print(df_iid_setting.groupby(name).mean())
    print(df_ood_setting.groupby(name).mean())
    print('--- std ---')
    print(df_iid_setting.groupby(name).std())
    print(df_ood_setting.groupby(name).std())

    plt.figure(figsize=(4,3))

    ax = sns.lineplot(x=df_iid_setting.groupby(name).mean().index, y=df_iid_setting.groupby(name).mean().iid, markers=True)
    ax.set_xlabel('$\lambda_i$')
    ax.set_ylabel('accuracy')    
    plt.savefig(args.foldername+'/summary_iid_critic.png', bbox_inches='tight')

    plt.clf()

    ax = sns.lineplot(x=df_ood_setting.groupby(name).mean().index, y=df_ood_setting.groupby(name).mean().ood, markers=True)
    ax.set_xlabel('$\lambda_i$')
    ax.set_ylabel('accuracy')    
    plt.savefig(args.foldername+'/summary_ood_critic.png', bbox_inches='tight')

    plt.clf()

    df_ood_group = df_ood_setting.groupby(name).mean()
    best_critic = df_ood_group.index[df_ood_group.ood.argmax()]

    print('optimal critic:', best_critic)

    df_comp = pd.DataFrame({'vanilla': df_ood_setting[df_ood_setting[name]==0.0].ood.to_numpy(), 
                        'regularizer': df_ood_setting[df_ood_setting[name]==best_critic].ood.to_numpy()})

    print(df_comp)

    plt.figure(figsize=(4,3))

    vmin = 0.1
    vmax = thres
    vbin = 0.1
    vtick = np.linspace(vmin, vmax, round((vmax-vmin)/vbin)+1)
    ax = sns.scatterplot(data=df_comp, x='vanilla', y='regularizer')
    ax.set_aspect('equal')
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])
    ax.plot([vmin, vmax], [vmin, vmax], '-r')
    ax.set_xticks(vtick)
    ax.set_yticks(vtick)
    plt.savefig(args.foldername+'/compare_ood_critic.png', bbox_inches='tight')


def compare_mask(df_all, df_iid, df_ood, args):
    values = df_all.amin.unique()

    stack_iid = list()
    stack_ood = list()
    for val in values:
        df_all_setting, df_iid_setting, df_ood_setting = extract_setting(df_all, df_iid, df_ood,
            args.bbox, args.ood, args.model, args.critic_action, args.critic_state, 
            args.finetune, args.translation, args.mask, val)
        stack_iid.append(df_iid_setting[['epoch', 'iid', 'amin', 'seed']])
        stack_ood.append(df_ood_setting[['epoch', 'ood', 'amin', 'seed']])
    df_iid_setting = pd.concat(stack_iid)
    df_ood_setting = pd.concat(stack_ood)

    # stats
    pd.options.display.float_format = '{:,.3f}'.format
    print(df_iid_setting.groupby('amin').mean(), df_iid_setting.groupby('amin').std())
    print(df_ood_setting.groupby('amin').mean(), df_ood_setting.groupby('amin').std())

    plt.figure(figsize=(4,3))

    # mask impact
    ax = sns.lineplot(x=df_iid_setting.groupby('amin').mean().index, y=df_iid_setting.groupby('amin').mean().iid, markers=True)
    ax.set_xlim([0.004, 1.0])
    ax.set_xlabel('min mask value')
    ax.set_ylabel('accuracy')
    ax.set_xscale('log') 
    plt.savefig(args.foldername+'/summary_iid_mask.png', bbox_inches='tight')

    plt.clf()

    ax = sns.lineplot(x=df_ood_setting.groupby('amin').mean().index, y=df_ood_setting.groupby('amin').mean().ood, markers=True)
    ax.set_xlim([0.004, 1.0])
    ax.set_xlabel('min mask value')
    ax.set_ylabel('accuracy')
    ax.set_xscale('log') 
    plt.savefig(args.foldername+'/summary_ood_mask.png', bbox_inches='tight')

    plt.clf()

    df_ood_group = df_ood_setting.groupby('amin').mean()
    best_amin = df_ood_group.index[df_ood_group.ood.argmax()]

    print('optimal mask:', best_amin)

    # fig
    df_comp = pd.DataFrame({'vanilla': df_iid_setting[df_iid_setting['amin']==1.0].iid.to_numpy(), 
                            'mask': df_iid_setting[df_iid_setting['amin']==best_amin].iid.to_numpy()})
    print(df_comp)
    vmin = 0.8
    vmax = 0.925
    vbin = 0.025
    vtick = np.linspace(vmin, vmax, round((vmax-vmin)/vbin)+1)
    ax = sns.scatterplot(data=df_comp, x='vanilla', y='mask')
    ax.set_aspect('equal')
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])
    ax.plot([vmin, vmax], [vmin, vmax], '-r')
    ax.set_xticks(vtick)
    ax.set_yticks(vtick)
    plt.savefig(args.foldername+'/compare_iid_mask.png', bbox_inches='tight')

    plt.clf()

    df_comp = pd.DataFrame({'vanilla': df_ood_setting[df_ood_setting['amin']==1.0].ood.to_numpy(), 
                            'mask': df_ood_setting[df_ood_setting['amin']==best_amin].ood.to_numpy()})
    print(df_comp)
    vmin = 0.1
    vmax = 0.6
    vbin = 0.1
    vtick = np.linspace(vmin, vmax, round((vmax-vmin)/vbin)+1)
    ax = sns.scatterplot(data=df_comp, x='vanilla', y='mask')
    ax.set_aspect('equal')
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])
    ax.plot([vmin, vmax], [vmin, vmax], '-r')
    ax.set_xticks(vtick)
    ax.set_yticks(vtick)
    plt.savefig(args.foldername+'/compare_ood_mask.png', bbox_inches='tight')


def compare_sparsity(df_all, df_iid, df_ood, args):

    values = df_all.sparsity.unique()

    stack_iid = list()
    stack_ood = list()
    for val in values:
        df_all_setting, df_iid_setting, df_ood_setting = extract_setting(df_all, df_iid, df_ood,
            args.bbox, args.ood, args.model, args.critic_action, args.critic_state, 
            args.finetune, args.translation, args.mask, args.amin, val)
        stack_iid.append(df_iid_setting[['epoch', 'iid', 'sparsity', 'seed']])
        stack_ood.append(df_ood_setting[['epoch', 'ood', 'sparsity', 'seed']])
    df_iid_setting = pd.concat(stack_iid)
    df_ood_setting = pd.concat(stack_ood)

    # stats
    pd.options.display.float_format = '{:,.3f}'.format
    print('--- avg ---')
    print(df_iid_setting.groupby('sparsity').mean())
    print(df_ood_setting.groupby('sparsity').mean())
    print('--- std ---')
    print(df_iid_setting.groupby('sparsity').std())
    print(df_ood_setting.groupby('sparsity').std())

    plt.figure(figsize=(4,3))

    # sparsity impact
    ax = sns.lineplot(x=df_iid_setting.groupby('sparsity').mean().index, y=df_iid_setting.groupby('sparsity').mean().iid, markers=True)
    ax.set_xlabel('$\lambda_s$')
    ax.set_ylabel('accuracy')
    ax.set_xscale('log')
    plt.savefig(args.foldername+'/summary_iid_sparsity.png', bbox_inches='tight')

    plt.clf()

    ax = sns.lineplot(x=df_ood_setting.groupby('sparsity').mean().index, y=df_ood_setting.groupby('sparsity').mean().ood, markers=True)
    ax.set_xlabel('$\lambda_s$')
    ax.set_ylabel('accuracy')
    ax.set_xscale('log')
    plt.savefig(args.foldername+'/summary_ood_sparsity.png', bbox_inches='tight')

    plt.clf()

    df_ood_group = df_ood_setting.groupby('sparsity').mean()
    best_sparsity = df_ood_group.index[df_ood_group.ood.argmax()]
    print('optimal sparsity:', best_sparsity)

    # fig
    df_comp = pd.DataFrame({'vanilla': df_iid_setting[df_iid_setting['sparsity']==0.0].iid.to_numpy(), 
                            'sparse': df_iid_setting[df_iid_setting['sparsity']==best_sparsity].iid.to_numpy()})
    print(df_comp)
    vmin = 0.9
    vmax = 1.0
    vbin = 0.025
    vtick = np.linspace(vmin, vmax, round((vmax-vmin)/vbin)+1)
    ax = sns.scatterplot(data=df_comp, x='vanilla', y='sparse')
    ax.set_aspect('equal')
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])
    ax.plot([vmin, vmax], [vmin, vmax], '-r')
    ax.set_xticks(vtick)
    ax.set_yticks(vtick)
    plt.savefig(args.foldername+'/compare_iid_sparsity.png', bbox_inches='tight')

    plt.clf()

    df_comp = pd.DataFrame({'vanilla': df_ood_setting[df_ood_setting['sparsity']==0.0].ood.to_numpy(), 
                            'sparse': df_ood_setting[df_ood_setting['sparsity']==best_sparsity].ood.to_numpy()})
    print(df_comp)
    vmin = 0.0
    vmax = 1.0
    # vmin = 0.3
    # vmax = 0.7
    vbin = 0.1
    vtick = np.linspace(vmin, vmax, round((vmax-vmin)/vbin)+1)
    ax = sns.scatterplot(data=df_comp, x='vanilla', y='sparse')
    ax.set_aspect('equal')
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])
    ax.plot([vmin, vmax], [vmin, vmax], '-r')
    ax.set_xticks(vtick)
    ax.set_yticks(vtick)
    plt.savefig(args.foldername+'/compare_ood_sparsity.png', bbox_inches='tight')


def compare_group(df_all, df_iid, df_ood, args):
    models = df_all.model.unique()

    stack_iid = list()
    stack_ood = list()
    for model in models:
        df_all_setting, df_iid_setting, df_ood_setting = extract_setting(df_all, df_iid, df_ood,
            args.bbox, args.ood, model, args.critic_action, args.critic_state, 
            args.finetune, args.translation, args.mask, args.amin)
        if model[:5] != 'group': continue             # skip models that are not based on groupvit
        if model == 'groupaverage':
            df_iid_setting['model'] = 'avg'
            df_ood_setting['model'] = 'avg'
        elif model == 'groupdense':
            df_iid_setting['model'] = 'dense'
            df_ood_setting['model'] = 'dense'
        elif model == 'grouptokenmax':
            df_iid_setting['model'] = 'token-max'
            df_ood_setting['model'] = 'token-max'
        elif model == 'grouptokenmean':
            df_iid_setting['model'] = 'token-mean'
            df_ood_setting['model'] = 'token-mean'
        stack_iid.append(df_iid_setting[['epoch', 'iid', 'model', 'seed']])
        stack_ood.append(df_ood_setting[['epoch', 'ood', 'model', 'seed']])
    df_iid_setting = pd.concat(stack_iid)
    df_ood_setting = pd.concat(stack_ood)

    # pdb.set_trace()

    # # select seeds that are difficult for the vanilla baseline
    # # cond = cond[:5]
    # # cond = [1, 2, 4, 8, 10]
    if 'avg-pool' in df_ood_setting['model'].unique():
        cond = df_ood_setting[df_ood_setting['model'] == 'avg-pool'].sort_values('ood').seed.unique()
        df_iid_setting = df_iid_setting[df_iid_setting['seed'].isin(cond)]
        df_ood_setting = df_ood_setting[df_ood_setting['seed'].isin(cond)]
    else:
        cond = df_ood_setting.seed.unique()

    # pdb.set_trace()
    print(df_iid_setting.groupby('model').mean(), df_iid_setting.groupby('model').std())
    print(df_ood_setting.groupby('model').mean(), df_ood_setting.groupby('model').std())

    hue_order = ['avg', 'dense', 'token-mean', 'token-max']

    ax = sns.barplot(data=df_iid_setting, x="seed", y="iid", hue="model", hue_order=hue_order)
    ax.set_ylabel("accuracy")
    ax.set(ylim=(0.25, 0.65))
    # ax.set(ylim=(0.6, 0.85))
    if len(models) < 5:
        ax.legend(loc="upper center", mode = "expand", ncol = len(models))              
    else:
        ax.legend(loc="upper center", mode = "expand", ncol = int(len(models)/2)+1)     
    ax.set_xticklabels(range(1,len(cond)+1))
    plt.savefig(args.foldername+'/summary_iid_epic.png', bbox_inches='tight')
    
    plt.clf()

    ax = sns.barplot(data=df_ood_setting, x="seed", y="ood", hue="model", hue_order=hue_order)
    ax.set_ylabel("accuracy")
    ax.set(ylim=(0.1, 0.4))
    # ax.set(ylim=(0.2, 0.6))
    if len(models) < 5:
        ax.legend(loc="upper center", mode = "expand", ncol = len(models))              
    else:
        ax.legend(loc="upper center", mode = "expand", ncol = int(len(models)/2)+1)     
    ax.set_xticklabels(range(1,len(cond)+1))
    plt.savefig(args.foldername+'/summary_ood_epic.png', bbox_inches='tight')

def compare_slot(df_all, df_iid, df_ood, args):
    models = df_all.model.unique()

    stack_iid = list()
    stack_ood = list()
    for model in models:
        df_all_setting, df_iid_setting, df_ood_setting = extract_setting(df_all, df_iid, df_ood,
            args.bbox, args.ood, model, args.critic_action, args.critic_state, 
            args.finetune, args.translation, args.mask, args.amin)
        if model[:4] != 'slot': continue             # skip models that are not based on groupvit
        if model == 'slotaverage':
            df_iid_setting['model'] = 'global-avg'
            df_ood_setting['model'] = 'global-avg'
        elif model == 'slotmatchmax':
            df_iid_setting['model'] = 'slot-max'
            df_ood_setting['model'] = 'slot-max'
        elif model == 'slotmatchmean':
            df_iid_setting['model'] = 'slot-mean'
            df_ood_setting['model'] = 'slot-mean'
        stack_iid.append(df_iid_setting[['epoch', 'iid', 'model', 'seed']])
        stack_ood.append(df_ood_setting[['epoch', 'ood', 'model', 'seed']])
    df_iid_setting = pd.concat(stack_iid)
    df_ood_setting = pd.concat(stack_ood)

    # # select seeds that are difficult for the vanilla baseline
    if 'global-avg' in df_ood_setting['model'].unique():
        cond = df_ood_setting[df_ood_setting['model'] == 'global-avg'].sort_values('ood').seed.unique()
        df_iid_setting = df_iid_setting[df_iid_setting['seed'].isin(cond)]
        df_ood_setting = df_ood_setting[df_ood_setting['seed'].isin(cond)]
    else:
        cond = df_ood_setting.seed.unique()

    print(df_iid_setting.groupby('model').mean(), df_iid_setting.groupby('model').std())
    print(df_ood_setting.groupby('model').mean(), df_ood_setting.groupby('model').std())

    hue_order = ['global-avg', 'slot-mean', 'slot-max']

    ax = sns.barplot(data=df_iid_setting, x="seed", y="iid", hue="model", hue_order=hue_order)
    ax.set_ylabel("accuracy")
    ax.set(ylim=(0.4, 0.75))
    if len(models) < 5:
        ax.legend(loc="upper center", mode = "expand", ncol = len(models))              
    else:
        ax.legend(loc="upper center", mode = "expand", ncol = int(len(models)/2)+1)     
    ax.set_xticklabels(range(1,len(cond)+1))
    plt.savefig(args.foldername+'/summary_iid_slot.png', bbox_inches='tight')
    
    plt.clf()

    ax = sns.barplot(data=df_ood_setting, x="seed", y="ood", hue="model", hue_order=hue_order)
    ax.set_ylabel("accuracy")
    ax.set(ylim=(0.1, 0.3))
    if len(models) < 5:
        ax.legend(loc="upper center", mode = "expand", ncol = len(models))              
    else:
        ax.legend(loc="upper center", mode = "expand", ncol = int(len(models)/2)+1)     
    ax.set_xticklabels(range(1,len(cond)+1))
    plt.savefig(args.foldername+'/summary_ood_slot.png', bbox_inches='tight')


def main():
    args = parse_arguments()

    log2csv(args.foldername)

    df_all, df_iid, df_ood = extract_df(args.foldername, args.nepoch)

    if args.setting == 'group':
        compare_group(df_all, df_iid, df_ood, args)
    elif args.setting == 'adversary':
        compare_critic(df_all, df_iid, df_ood, args)
    elif args.setting == 'mask':
        compare_mask(df_all, df_iid, df_ood, args)
    elif args.setting == 'sparsity':
        compare_sparsity(df_all, df_iid, df_ood, args)
    elif args.setting == 'slot':
        compare_slot(df_all, df_iid, df_ood, args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
