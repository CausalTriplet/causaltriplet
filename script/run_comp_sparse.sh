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

	for ((SEED=1;SEED<=10;SEED++))
	do
		python main.py --path_data=$DATADIR --ood=comp --model=resnet18 --num_workers=64 --batch_size=128 --print_freq=1000 --bbox --finetune --epochs=50 --sparse=0.0 --seed=${SEED}
		python main.py --path_data=$DATADIR --ood=comp --model=resnet18 --num_workers=64 --batch_size=128 --print_freq=1000 --bbox --finetune --epochs=50 --sparse=0.001 --seed=${SEED}
		python main.py --path_data=$DATADIR --ood=comp --model=resnet18 --num_workers=64 --batch_size=128 --print_freq=1000 --bbox --finetune --epochs=50 --sparse=0.01 --seed=${SEED}
		python main.py --path_data=$DATADIR --ood=comp --model=resnet18 --num_workers=64 --batch_size=128 --print_freq=1000 --bbox --finetune --epochs=50 --sparse=0.1 --seed=${SEED}
		python main.py --path_data=$DATADIR --ood=comp --model=resnet18 --num_workers=64 --batch_size=128 --print_freq=1000 --bbox --finetune --epochs=50 --sparse=1.0 --seed=${SEED}
	done

fi