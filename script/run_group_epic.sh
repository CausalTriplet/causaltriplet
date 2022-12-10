#!/usr/bin/env bash

# ------ variables ------

if [ "$local" = true ] ; then
    DATADIR=data
else
	DATADIR=/storage/datasets/epic
fi

echo $DATADIR

# ------ python ------

if [ "$local" = true ] ; then

	echo "local"

else

	echo "remote"

	for ((SEED=1;SEED<=10;SEED++))
	do
		python main.py --dataset=epickitchens --path_data=$DATADIR --ood=noun --model=groupaverage --lr=0.001 --num_workers=64 --batch_size=128 --print_freq=1000 --epochs=200 --seed=${SEED}
		python main.py --dataset=epickitchens --path_data=$DATADIR --ood=noun --model=groupdense --lr=0.001 --num_workers=64 --batch_size=128 --print_freq=1000 --epochs=200 --seed=${SEED}
		python main.py --dataset=epickitchens --path_data=$DATADIR --ood=noun --model=grouptokenmean --lr=0.001 --num_workers=64 --batch_size=128 --print_freq=1000 --epochs=200 --seed=${SEED}
		python main.py --dataset=epickitchens --path_data=$DATADIR --ood=noun --model=grouptokenmax --lr=0.001 --num_workers=64 --batch_size=128 --print_freq=1000 --epochs=200 --seed=${SEED}
	done

fi