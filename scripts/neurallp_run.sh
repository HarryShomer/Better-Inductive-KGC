#!/bin/bash

dataset=$1
device=$2

cd ../src/nbfnet

for i in {1..5}; do
    printf "Seed=${i}\n"
    python run.py -c config/neurallp/${dataset}.yaml --gpus [$device] --seed $i --save_as "NeuralLP_${dataset}_Seed-${i}"  
done
