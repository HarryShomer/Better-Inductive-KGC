# Generatee New Datasets

The set of new inductive datasets that were used in the study, can be reproduced by running the script `scripts/generate_new_splits.sh`. 

A custom inductive dataset can be created by running the `src/generate_new_splits.py` script. Multiple options exist, including:
- `--alg`: The clustering algorithm. This includes spectral clustering (specified by `spectral`) and louvain (specificed by, you guessed it, `louvain`).
- ``--num-clusters``: The \# of clusters to consider. This option is only considered when using spectral clustering. 
- `--type`: The type of inductive task. Either `E` for (E) and `ER` for (E, R).
- `--lcc`: A flag that indicates if we should take the largest connected component for the dataset. This is recommended. 
- `save-as`: Name of folder to save data to. Will be saved to `new_data/{name}/`

You must manually choose which graphs to choose for training and testing. In order to decide which to choose, we first run the script to print the various options. Once chosen, we run it again with which graphs we want. To list the different candidate graphs, we must pass the `--print-candidates` argument. This will give you the relevant statistics for the top k clusters created (default is 5). The number to consider can be modified via `--candidates`. An example is given below:
```
python generate_new_splits.py --dataset CoDEx-m --num-clusters 10 --lcc --print-candidates --candidates 6
```
Each candidate graph printed will have a corresponding ID number. To choose which graphs you want for training and test you must run the script again but this time specifying which graphs to choose. **Note that the order matters**. The first graph is considered the training graph, while the rest are for inference. For example, after examining the candidates from the previous command, let's say we want to choose graph 7 for training and 6 and 5 for inference. This is done by running:
```
python generate_new_splits.py --dataset CoDEx-m --num-clusters 10 --choose-graphs 7 6 5 --save-as codex_m_E --lcc
```