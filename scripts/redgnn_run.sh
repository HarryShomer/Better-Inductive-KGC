#!/bin/bash

dataset=$1
lr=$2
drop=$3
num_test=$4
device=$5

cd ../src/RED-GNN

for i in {1..5}; do
    printf "Seed=${i}\n"
    CUDA_VISIBLE_DEVICES=$device python -W ignore train.py --data_path=../../new_data/${dataset} \
                          --lr $lr --dropout $drop --num-test $num_test --seed $i
done

