import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob

import csv
import re

import pdb

sns.set_theme()
pd.options.display.float_format = '{:,.2f}'.format
plt.rcParams["figure.dpi"] = 140


def analyze_log(f):
    stat = list()
    for line in f:
        if len(line) > 5 and line[:5] == 'epoch':
            raw = re.split('#|\(|\)|=|\n|,| ', line)
            num = [s for s in raw if re.match('^[-+]?[0-9.]+$', s)]
            stat.append(num)
    return stat


def write_stats(stat, f):
    out = csv.writer(f)
    out.writerow(["epoch", "train", "iid", "ood", "val", "obj_action", "obj_state", "sparsity"])
    out.writerows(stat)


def log2csv(foldername):
    flist = glob.glob(foldername+'/*.log')
    flist.sort()

    for fname in flist:
        with open(fname) as input_file:
            stat = analyze_log(input_file)
        with open(fname[:-3] + 'csv', 'w') as output_file:
            write_stats(stat, output_file)


def extract_csv(fname):
    df = pd.read_csv(fname, header=0, sep=",", skipinitialspace=True)
    parse = fname.split('/')[2].split('_')

    # print(parse)
    df['split'] = parse[0]
    df['data'] = int(parse[1])
    df['model'] = parse[2]
    df['hdim'] = int(parse[3])
    df['tran'] = float(parse[5])
    df['critic_action'] = float(parse[7])
    df['critic_state'] = float(parse[8])
    df['linear'] = (parse[10]) == 'True'
    df['mask'] = (parse[12]) == 'True'
    df['bbox'] = (parse[14]) == 'True'
    df['encoder'] = (parse[16]) == 'True'

    if len(parse) > 18:
        df['ckpt'] = (parse[18])

    if len(parse) > 24:     # since sept
        df['sparsity'] = float(parse[20])
        df['amin'] = float(parse[22])
    elif len(parse) > 21:
        df['amin'] = float(parse[20])

    df['seed'] = int(parse[-1][:-4])
    return df


def plot_curve(df, title='none'):
    #     plt.figure()
    dfm = df.melt('epoch', var_name='setting', value_name='accuracy')
    snsplot = sns.lineplot(data=dfm, x='epoch', y='accuracy', hue='setting')
    snsplot.set(xlim=[0, 30], ylim=[0.0, 1.05], title=title)
    snsplot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


def extract_setting(df, sel_iid, sel_ood, bbox, split, model, critic_action, critic_state, encoder, tran, mask, amin, sparsity=-1, plot=False):
    df_setting = df[(df['bbox'] == bbox)
                    & (df['split'] == split)
                    & (df['model'] == model)
                    & (df['critic_action'] == critic_action)
                    & (df['critic_state'] == critic_state)
                    & (df['encoder'] == encoder)
                    & (df['tran'] == tran)
                    & (df['mask'] == mask)
                    & (df['amin'] == amin)
                    ]
    if sparsity >= 0:
        df_setting = df_setting[df_setting['sparsity'] == sparsity]

    df_filter = df_setting[['epoch', 'train', 'iid', 'ood', 'val']]
    # if plot:
    #     plt.figure()
    #     plot_curve(df_filter, f'critic: {critic}   ood: {split}   train size: {data}   model: {model}     bbox: {bbox}')

    df_filter = df_setting[['epoch', 'obj_action', 'obj_state']]
    # if plot:
    #     plt.figure()
    #     plot_curve(df_filter, f'critic: {critic}   ood: {split}   train size: {data}   model: {model}     bbox: {bbox}')

    sel_iid = sel_iid.sort_values('seed')
    sel_iid_setting = sel_iid[(sel_iid['bbox'] == bbox)
                              & (sel_iid['split'] == split)
                              & (sel_iid['model'] == model)
                              & (sel_iid['critic_action'] == critic_action)
                              & (sel_iid['critic_state'] == critic_state)
                              & (sel_iid['encoder'] == encoder)
                              & (sel_iid['tran'] == tran)
                              & (sel_iid['mask'] == mask)
                              & (sel_iid['amin'] == amin)
                              ]
    if sparsity >= 0:
        sel_iid_setting = sel_iid_setting[sel_iid_setting['sparsity'] == sparsity]

    # if plot:
    #     sel_iid_setting.plot.bar(x='seed', y='iid', ylim=[0.0, 1.0], title=f'critic_action={critic_action}    critic_state={critic_state}')

    sel_ood = sel_ood.sort_values('seed')
    sel_ood_setting = sel_ood[(sel_ood['bbox'] == bbox)
                              & (sel_ood['split'] == split)
                              & (sel_ood['model'] == model)
                              & (sel_ood['critic_action'] == critic_action)
                              & (sel_ood['critic_state'] == critic_state)
                              & (sel_ood['encoder'] == encoder)
                              & (sel_ood['tran'] == tran)
                              & (sel_ood['mask'] == mask)
                              & (sel_ood['amin'] == amin)
                              ]
    if sparsity >= 0:
        sel_ood_setting = sel_ood_setting[sel_ood_setting['sparsity'] == sparsity]

    # if plot:
    #     sel_ood_setting.plot.bar(x='seed', y='ood', ylim=[0.0, 1.0], title=f'critic_action={critic_action}    critic_state={critic_state}')

    return df_setting, sel_iid_setting, sel_ood_setting


def extract_df(foldername, nepoch):
    pattern = f'{foldername}/*.csv'
    filelist = glob.glob(pattern)
    filelist.sort()
    # print(filelist)

    stack_all = list()
    stack_iid = list()
    stack_ood = list()
    for fname in filelist:
        df = extract_csv(fname)
        df = df[df['epoch'] < nepoch]
        stack_all.append(df)
        stack_iid.append(df.iloc[df.iid.idxmax()])
        stack_ood.append(df.iloc[df.val.idxmax()])
    df_all = pd.concat(stack_all)
    df_iid = pd.concat(stack_iid, axis=1).transpose()
    df_ood = pd.concat(stack_ood, axis=1).transpose()
    return df_all, df_iid, df_ood
