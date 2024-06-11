#!/bin/bash

cd ../src

# (E) datasets
python generate_new_splits.py --dataset CoDEx-m --num-clusters 5 --choose-graphs 4 3 --save-as codex_m_E --lcc
python generate_new_splits.py --dataset WN18RR --num-clusters 50 --choose-graphs 49 40 48 --save-as wn18rr_E
python generate_new_splits.py --dataset HetioNet --num-clusters 25 --choose-graphs 21 19 20  --save-as hetionet_E --lcc

# (E, R) datasets
python generate_new_splits.py --dataset FB15k-237 --alg louvain --choose-graphs 2 10 3 --type ER --lcc --save-as fb15k-237_ER
python generate_new_splits.py --dataset CoDEx-m  --num-clusters 6 --choose-graphs 4 5 2 --save-as codex_m_ER  --type ER --lcc