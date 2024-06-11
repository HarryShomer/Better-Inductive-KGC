#!/bin/bash

cd ../src/nbfnet

device=$1

python run_ppr.py -c config/new/wn18rr_e.yaml --gpus [$device]
python run_ppr.py -c config/new/codex_m_e.yaml --gpus [$device]
python run_ppr.py -c config/new/hetionet_e.yaml --gpus [$device] --eps 1e-5
python run_ppr.py -c config/new/fb15k237_er.yaml --gpus [$device]
python run_ppr.py -c config/new/codex_m_er.yaml --gpus [$device]