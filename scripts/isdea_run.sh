#!/bin/bash

dataset=$1
lr=$2
device=$3
folder="/egr/research-dselab/shomerha/kg_ppr/new_data/"

cd ../src/ISDEA_PLUS

# When we have best hyperparameters
for i in {1..5}; do
    python src/main_new.py --exp_name "${dataset}_${i}" --dataset_folder $folder  --dataset $dataset --mode train  \
                           --epoch 200 --valid_epoch 20 --seed $i --device $device --lr $lr
done

