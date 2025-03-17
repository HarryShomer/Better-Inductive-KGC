# Better Inductive KGC Datasets

Official code for the paper ["Towards Better Benchmark Datasets for Inductive Knowledge Graph Completion"](https://arxiv.org/abs/2406.11898).


## Installation

Please see [install.md](./install.md) for how to install the code and the proper package requirements. We note that some methods, like ULTRA and NodePiece, require their own special environment.


## Data 

**NOTE**: Currently, the entity IDs aren't guranteed to map to the corresponding ID in the original dataset. This has no effect on the results of any of the methods used in the paper. This will have an effect if you want to map the entities back to their original natural language description.

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

For a detailed overview of the dataset, see the [datasheet.md](./datasheet.md) file.

## Reproduce Results

Please see [run.md](./run.md) for how to reproduce the results in the paper. 


## Generate the New Datasets

Please see [generate_new.md](./generate_new.md) for how to regenerate the new datasets created in the paper. We further give instructions for how to generate your own new inductive datasets.


## Cite

```
@article{shomer2024ppr,
      title={Towards Better Benchmark Datasets for Inductive Knowledge Graph Completion}, 
      author={Harry Shomer and Jay Revolinsky and Jiliang Tang},
      journal={arXiv preprint arXiv:2406.11898},
      year={2024}
}
```
