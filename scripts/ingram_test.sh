#!/bin/bash

###########################
# For evaluating on InGram
###########################

dataset=$1
lr=$2
nle=$3
num_test=$4
device=$5

cd ../src/InGram

for i in {1..5}; do
    for ((j = 0; j < num_test; j++)); do
        printf ">>> Seed=$i, Graph=$j\n"
        CUDA_VISIBLE_DEVICES=$device python test.py --data_name $dataset -lr $lr -nle $nle --seed $i --test-graph $j
    done
done
