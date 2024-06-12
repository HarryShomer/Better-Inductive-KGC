# Better Inductive KGC Datasets

Official code for the paper ["Towards Better Benchmark Datasets for Inductive Knowledge Graph Completion"]().


## Installation

Please see the [install.md](./install.md) for how to install the code and proper requirements. 


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

Please see the [generate_new.md](./generate_new.md) for how to regenerate the new datasets created in the paper. We further give instructions for how to generate your own new inductive datasets.


## Cite


