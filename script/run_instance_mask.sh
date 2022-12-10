#!/usr/bin/env bash

# ------ variables ------

if [ "$local" = true ] ; then
	DATADIR=data
else
	DATADIR=/storage/datasets/thor
fi

echo $DATADIR

# ------ python ------

if [ "$local" = true ] ; then

	echo "local"

else

	echo "remote"

	# --- non-mask baseline ---

	python main.py --path_data=$DATADIR --ood=noun --model=resnet18 --num_workers=64 --batch_size=128 --print_freq=1000 --finetune --mask --amin=1.0 --epochs=50

	# --- approx. object-centric ---

	python main.py --path_data=$DATADIR --ood=noun --model=resnet18 --num_workers=64 --batch_size=128 --print_freq=1000 --finetune --mask --amin=0.005 --epochs=50
	python main.py --path_data=$DATADIR --ood=noun --model=resnet18 --num_workers=64 --batch_size=128 --print_freq=1000 --finetune --mask --amin=0.01 --epochs=50
	python main.py --path_data=$DATADIR --ood=noun --model=resnet18 --num_workers=64 --batch_size=128 --print_freq=1000 --finetune --mask --amin=0.05 --epochs=50
	python main.py --path_data=$DATADIR --ood=noun --model=resnet18 --num_workers=64 --batch_size=128 --print_freq=1000 --finetune --mask --amin=0.1 --epochs=50
	python main.py --path_data=$DATADIR --ood=noun --model=resnet18 --num_workers=64 --batch_size=128 --print_freq=1000 --finetune --mask --amin=0.5 --epochs=50

fi