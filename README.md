# HeaRT

Official code for the paper ["Towards Better Benchmark Datasets for Inductive Knowledge Graph Completion"]().


## Installation

Please see the [install.md](./install.md) for how to install the proper requirements. 


## Data

All the data can be found in the `new_data` folder. Each new dataset is further categorized into their own folder (e.g., `new_data/wn18rr_E`). The data is split into the following files:
- `train_graph`: Contain the triples used during training
- `valid_samples`: Contain the triples used during validation. Note that these samples correspond to the training graph.
- `test_{i}_graph`: The triples in inference graph `i`.  
- `test_{i}_samples`: The test triples for the inference graph `i`.

We follow the common data storage convention and have each line contain 1 triple, with the head/rel/tail separated by a space. For example, a single file could be read by pandas with:
```
import pandas as pd 
df = pd.read_csv("train_graph.txt", header=None, names=["head", "rel", "tail"], delimiter=" ")
```


## Reproduce Results

Please see the [run.md](./run.md) for how to reproduce the results in the paper. 


## Generate the New Datasets

The set of new inductive datasets, that were used in the study, can be reproduced by running the script `scripts/generate_new_splits.sh`. 

A custom inductive dataset can be created by running the `src/generate_new_splits.py` script. Multiple options exist, including:
- `--alg`: The clustering algorithm. This includes spectral clustering (specified by `spectral`) and louvain (specificed by, you guessed it, `louvain`).
- ``--num-clusters``: The \# of clusters to consider. This option is only considers when using spectral clustering. 
- `--type`: Specificy the type of inductive task. Either `E` for (E) and `ER` for (E, R).
- `--lcc`: A flag that indicates if we should take the largest common component for the dataset. This is recommended. 
- `save-as`: Name of folder to save data to. Will be saved to `new_data/{name}/`

You must manually choose which graphs to choose for training and testing. In order to decide which to choose, we first run the script to print the various options. Once chosen, we run it again with which graphs we want. To list the different candidate graphs, we must pass the `--print-candidates` argument. This will give you the relevant statistics for the top k clusters created (default is 5). The number to consider can be modified via `--candidates`. An example is given below:
```
python generate_new_splits.py --dataset CoDEx-m --num-clusters 10 --lcc --print-candidates --candidates 6
```
Each candidate graph printed will have a corresponding ID number. To choose which graphs you want for training and test you must run the script again but this time specifying which graphs to choose. **Note that the order matters**. The first graph is considered the training graph, while the rest are for inference. For example, after examining the candidates from the previous command, let's say we want to choose graph 7 for training and 6 and 5 for inference. This is done by running:
```
python generate_new_splits.py --dataset CoDEx-m --num-clusters 10 --choose-graphs 7 6 5 --save-as codex_m_E --lcc
```

## Cite


