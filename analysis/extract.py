import csv
import re
import glob

import pdb


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
    out.writerow(["epoch", "train", "iid", "ood", "val", "obj_action", "obj_state"])
    out.writerows(stat)


def main(foldername):
    flist = glob.glob(foldername+'/*.log')
    flist.sort()

    for fname in flist:
        with open(fname) as input_file:
            stat = analyze_log(input_file)
        with open(fname[:-3] + 'csv', 'w') as output_file:
            write_stats(stat, output_file)


if __name__ == '__main__':
    # main(r'save/log_comp_10000_0818')
    # main(r'save/log_noun_5000_0820')
    # main(r'save/log_noun_resnet50_0821')
    # main(r'save/log_tcn_5000_0821')
    # main(r'save/log_clip_5000_0822')
    # main(r'save/log_mask_5000_0822')
    # main(r'save/log_group_5000_0822')
    # main(r'save/log_mask_5000_0823')
    # main(r'save/log_comp_10000_0823')
    # main(r'save/log_comp_10k')
    # main(r'save/log_group_10000_0824')
    # main(r'save/log_comp_5000_0824')
    main(r'save/log_mask_5000_0824')
