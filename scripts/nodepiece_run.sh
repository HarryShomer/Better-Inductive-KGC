#!/bin/bash

dataset=$1
num_test=$2
sample_rels=$3
margin=$4
device=$5

cd ../src/NodePiece/inductive_lp

for i in {5..5}; do
    printf "Seed=${i}"
    CUDA_VISIBLE_DEVICES=$device python run_ilp.py -loss nssal -margin $margin -epochs 500 -lr 0.001 -data $dataset \
                                -sample_rels $sample_rels -negs 32 -gnn True -rp False -gnn_att True -gnn_layers 3 \
                                -path /egr/research-dselab/shomerha/kg_ppr/new_data  -num_test $num_test \
                                -pna False -residual True -jk False -seed $i
done