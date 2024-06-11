#!/bin/bash

dataset=$1
num_test=$2
ckpt=$3
device=$4

cd ../src/ULTRA

for ((i = 0; i < num_test; i++)); do
    printf ">>> Test Graph $i\n"
    python script/run.py -c config/new_splits.yaml --dataset $dataset --inf-graph $i --num-test $num_test --epochs 0 --bpe null --gpus [$device] --ckpt $ckpt
done