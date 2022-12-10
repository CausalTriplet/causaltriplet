#!/usr/bin/env bash

# ----- group structure -----

python analysis/analysis.py --foldername='logs/log_epic_0825' --setting='group' --ood='noun'

# ----- instance mask -----

python analysis/analysis.py --foldername='logs/log_mask_thor_0824' --setting='mask' --ood='noun' --model='resnet18' --mask --finetune

# ----- block disentanglement -----

python analysis/analysis.py --foldername='logs/log_comp_sparse_0912' --setting='sparsity' --ood='comp' --model='resnet18' --bbox --finetune --nepoch=20

# ----- slot structure -----

python analysis/analysis.py --foldername='logs/log_slot_thor_1208' --setting='slot' --ood='noun'
