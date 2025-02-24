#!/bin/bash

dataset=$1
inf_graph=$2
lr=$3
nle=$4
dev=$5
folder="/egr/research-dselab/shomerha/kg_ppr/new_data/"

cd ../src/DEq-InGram

for seed in {1..5}; do
    CUDA_VISIBLE_DEVICES=$dev python test.py --best --run_hash "Seed-${seed}" --data_name $dataset \
                                            --data_path $folder --exp "${dataset}_${lr}_${nle}" --mc 10 --inf_graph $inf_graph
done