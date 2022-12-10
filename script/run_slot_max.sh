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

	for ((SEED=1;SEED<=5;SEED++))
	do
	python main.py --path_data=$DATADIR --ood=noun --model=slotmatchmax --num_workers=32 --batch_size=200 --print_freq=5 --epochs=50 --ckpt=ckpt/slot.pt.tar --lr=0.002 --seed=${SEED}
	done

fi