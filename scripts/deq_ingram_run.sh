#!/bin/bash

dataset=$1
lr=$2
nle=$3
dev=$4
folder="/egr/research-dselab/shomerha/kg_ppr/new_data/"

cd ../src/DEq-InGram

for seed in {1..5}; do
    CUDA_VISIBLE_DEVICES=$dev python train.py --data_path $folder --data_name $dataset  \
                                              --exp "${dataset}_${lr}_${nle}" -lr $lr -nle $nle --seed $seed
done