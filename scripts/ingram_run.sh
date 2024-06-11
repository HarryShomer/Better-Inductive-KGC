#!/bin/bash

dataset=$1
lr=$2
nle=$3
device=$4

cd ../src/InGram

# When we have best hyperparameters
for i in {1..5}; do
     CUDA_VISIBLE_DEVICES=$device python train.py --data_name $dataset --seed $i -lr $lr -nle $nle
done

