import os
import torch
import pickle
import random
import argparse
import numpy as np
import networkx as nx
from torch_geometric.data import Data

import igraph as ig
from utils import calc_path_length, gen_k_negs_per_pos, calc_emd_dist


DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "generated")


def relabel_graph(edges):
    """
    Saved numbering doesn't reflect # of nodes. 
    
    Fix this

    Returns mapping to new IDs
    """
    old2newid = {}
    num_nodes = 0

    for h, _, t in edges:
        h, t = int(h), int(t)

        if h not in old2newid:
            old2newid[h] = num_nodes
            num_nodes += 1
        if t not in old2newid:
            old2newid[t] = num_nodes
            num_nodes += 1

    return old2newid


def get_data(rawdata, split="test"):
    """
    Get train/test data
    """
    graphname = "inf_graph" if split == "test" else "train"
    
    mapnewids = relabel_graph(rawdata[graphname])

    edge_index = torch.Tensor([(mapnewids[int(t[0])], mapnewids[int(t[2])]) for t in rawdata[graphname]]).t().long()
    edge_type = torch.Tensor([int(t[1]) for t in rawdata[graphname]]).t().long()

    num_nodes = torch.max(edge_index).item() + 1

    smplsname = "test" if split == "test" else "valid"
    std_ei, std_et = [], []
    for h, r, t in rawdata[f'{smplsname}_samples']:
        if int(h) in mapnewids and int(t) in mapnewids:
            std_ei.append((mapnewids[int(h)], mapnewids[int(t)]))
            std_et.append(int(r))
            
    target_edge_index = torch.Tensor(std_ei).t().long()
    target_edge_type = torch.Tensor(std_et).t().long()
    test_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                     target_edge_index=target_edge_index, target_edge_type=target_edge_type)

    return test_data



def detect_communities(G):
    """
    1. Louvain
    2. LP
    """
    lv = nx.community.louvain_communities(G, weight="weight", seed=42)
    # print(f'Louvain # = {len(lc)}')

    lp = nx.community.asyn_lpa_communities(G, weight="weight", seed=42)
    # print(f'LP # = {len([l for l in lp])}')

    return len(lv), len([l for l in lp])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="FB15k-237")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument("--num-negs", help="Negs per positive", type=int, default=25)
    parser.add_argument('--num-seeds', type=int, default=3)
    args = parser.parse_args()

    all_emd = []

    for s in range(1, args.num_seeds+1):
        # print(f">>> Seed={s}")
        with open(os.path.join(DATA_DIR, f"{args.dataset}_seed-{s}.pkl"), 'rb') as handle:
            sdata = pickle.load(handle)
        
        data = get_data(sdata, args.split)

        neg_samples = gen_k_negs_per_pos(data, args.num_negs)
        pos_samples = torch.cat((data.target_edge_index, data.target_edge_index[[1, 0]]), dim=-1).t().tolist()

        pos_sp = calc_path_length(data, pos_samples)
        neg_sp = calc_path_length(data, neg_samples)
        pos_sp, neg_sp = np.array(pos_sp), np.array(neg_sp)

        all_emd.append(calc_emd_dist(pos_sp, neg_sp))

    print(f"SP EMD = {np.mean(all_emd)}")


if __name__ == "__main__":
    main()
