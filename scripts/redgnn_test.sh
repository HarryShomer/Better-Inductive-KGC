#!/bin/bash

###########################
# For evaluating on RED-GNN
###########################

dataset=$1
num_test=$2
device=$3

cd ../src/RED-GNN

for i in {1..5}; do
    printf ">>> Seed=$i\n"
    CUDA_VISIBLE_DEVICES=$device python -W ignore test.py  --data_path=../../new_data/${dataset} --num-test $num_test --seed $i
done


