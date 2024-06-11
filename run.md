# Reproducing Results

We detail how to reproduce all the results for each method. We further show how to run each method for yourself. Note that for the neural methods, they are each run over 5 different random seeds.

By this point, you should have already installed the proper environments. If not, please see [install.md](./install.md) for instructions on how to do so. Unless otherwise specicied, you should use the `std_env` environment.


## Personalized PageRank (PPR)

This is done via `scripts/ppr_run.sh`. You must also pass the GPU to use when running:
```
cd scripts
bash ppr_run.sh 3  # Run with GPU 3
```

You can run it youself with the following where you must replace `DEVICE` with the integer GPU device being used.
```
cd ../src/nbfnet
python run_ppr.py -c config/new/wn18rr_e.yaml --gpus [DEVICE]
```

## NodePiece

**NOTE**: NodePiece requires a specific environment (see [install.md](./install.md)) to run correctly.

It can be run by the following where `<>` denotes a value that must be specified:
```
cd scripts
bash nodepiece_run.sh <DATASET> <NUM_TEST> <SAMPLE_RELS> <MARGIN> <DEVICE>
```

To replicate the exact results:
- WN18RR (E): `bash nodepiece_run.sh wn18rr_E 2 4 25 <DEVICE>`
- CoDEx-M (E): `bash nodepiece_run.sh codex_m_E 1 12 15 <DEVICE>`
- HetioNet (E): `bash nodepiece_run.sh hetionet_E 2 4 15 <DEVICE>`
- FB15k-237 (E, R): `bash nodepiece_run.sh fb15k-237_ER 2 12 15 <DEVICE>`
- CoDEx-M (E, R): `bash nodepiece_run.sh codex_m_ER 2 4 25 <DEVICE>`


## InGRAM

InGRAM contains separate scripts for training and testing.

It can be **trained** by the following, where `<>` denotes a value that must be specified:
```
cd scripts
bash ingram_run.sh <DATASET> <LEARNING_RATE> <NUM_ENTITY_LAYERS> <DEVICE>
```

To replicate the exact trained models:
- WN18RR (E): `bash ingram_run.sh wn18rr_E  1e-3 2 <DEVICE>`
- CoDEx-M (E): `bash ingram_run.sh codex_m_E 5e-4 4 <DEVICE>`
- HetioNet (E): `bash ingram_run.sh hetionet_E 5e-4 4 <DEVICE>`
- FB15k-237 (E, R): `bash ingram_run.sh fb15k-237_ER 1e-3 2 <DEVICE>`
- CoDEx-M (E, R): `bash ingram_run.sh codex_m_ER 5e-4 2 <DEVICE>`

To test each method, you must pass the same hyperparameters used for training, with an additional parameter for the \# of inference graphs.
```
bash ingram_test.sh <DATASET> <LEARNING_RATE> <NUM_ENTITY_LAYERS> <NUM_TEST> <DEVICE>
```

To replicate the test results:
- WN18RR (E): `bash ingram_test.sh wn18rr_E  1e-3 2 2 <DEVICE>`
- CoDEx-M (E): `bash ingram_test.sh codex_m_E 5e-4 4 1 <DEVICE>`
- HetioNet (E): `bash ingram_test.sh hetionet_E 5e-4 4 2 <DEVICE>`
- FB15k-237 (E, R): `bash ingram_test.sh fb15k-237_ER 1e-3 2 2 <DEVICE>`
- CoDEx-M (E, R): `bash ingram_test.sh codex_m_ER 5e-4 2 2 <DEVICE>`


## RED-GNN

To train and evaluate RED-GNN:
```
cd scripts
bash redgnn_run.sh <DATASET> <LEARNING_RATE> <DROPOUT> <NUM_TEST> <DEVICE>
```

To replicate the test results:
- WN18RR (E): `bash redgnn_run.sh wn18rr_E  5e-3 0.3 2 <DEVICE>`
- CoDEx-M (E): `bash redgnn_run.sh codex_m_E 5e-3 0.3 1 <DEVICE>`
- HetioNet (E): `bash redgnn_run.sh hetionet_E 5e-4 0.1 2 <DEVICE>`
- FB15k-237 (E, R): `bash redgnn_run.sh fb15k-237_ER 5e-3 0.3 2 <DEVICE>`
- CoDEx-M (E, R): `bash redgnn_run.sh codex_m_ER 5e-3 0.3 2 <DEVICE>`

## NBFNet

NBFNet differs from the others as the optimal hyperparameters reside in the config files located in the `src/nbfnet/config` directory. We note that this organization follows the same structure as their original implementation. 

To replicate the test results run the following. Note that the dataset here must match the corresponding name of the file in `src/nbfnet/config/new` for that dataset.
- WN18RR (E): `bash nbfnet_run.sh wn18rr_e <DEVICE> <SAVE_AS>`
- CoDEx-M (E): `bash nbfnet_run.sh codex_m_e <DEVICE> <SAVE_AS>`
- HetioNet (E): `bash nbfnet_run.sh hetionet_e <DEVICE> <SAVE_AS>`
- FB15k-237 (E, R): `bash nbfnet_run.sh fb15k237_er <DEVICE> <SAVE_AS>`
- CoDEx-M (E, R): `bash nbfnet_run.sh codex_m_er <DEVICE> <SAVE_AS>`

To modify the hyperparameters for a given dataset, see the config in `src/nbfnet/config/new`. Also, if you are adding your own dataset, please note that you must specify the correct information under the `dataset` header in the config file. For example:
```
dataset:
  class: <DATASET_NAME>  # Must match name of folder
  root: <DIRECTORY>  # Location when this is. You probably don't need to change this
  new: yes
  num_test: # Number of inference graphs
```

## ULTRA

**NOTE**: ULTRA requires a specific environment (see [install.md](./install.md)) to run correctly.

In the paper, we evaluate ULTRA under the 0-shot setting. Therefore, no training or fine-tuning is required. 

By default, ULTRA comes with several pre-trained models that were trained on multiple transductive datasets. However, since we generated new inductive datasets from some of those transductive datasets used (e.g., FB15k-237) there was the potential of test leakage. To account for this we trained our own verison of ULTRA that didn't include that dataset (when applicable). We found that this had a modest but negative impact on performance (see our paper for more details). These checkpoints are included in our repo.

ULTRA can be run with the following where the potential values of `CKPT` can be found in `src/ULTRA/ckpts/`.
```
cd scripts
bash ultra_run.sh <DATASET> <NUM_TEST> <CKPT> <DEVICE>
```

Here are the commands for replicating our results:
- WN18RR (E): `bash ultra_run.sh wn18rr_ER 2 ultra_3g_wo_wn18rr.pth <DEVICE>`
- CoDEx-M (E): `bash ultra_run.sh codex_m_E 1 ultra_3g_wo_codex.pth <DEVICE>`
- HetioNet (E): `bash ultra_run.sh hetionet_E 2 ultra_4g.pth <DEVICE>`
- FB15k-237 (E, R): `bash ultra_run.sh fb15k-237_ER 2 ultra_3g_wo_fb15k237.pth <DEVICE>`
- CoDEx-M (E, R): `bash ultra_run.sh codex_m_ER 2 ultra_3g_wo_codex.pth <DEVICE>`