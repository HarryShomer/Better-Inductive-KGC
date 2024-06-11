#!/bin/bash

dataset=$1
device=$2
save_as=$3

cd ../src/nbfnet

for i in {1..5}; do
    printf "Seed=${i}\n"
    python run.py -c config/new/${dataset}.yaml --gpus [$device] --seed $i --save_as "${save_as}_Seed-${i}"  
done
