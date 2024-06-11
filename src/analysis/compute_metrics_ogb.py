import os
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from torch_sparse import SparseTensor
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree, to_undirected
from torch_geometric.utils import to_undirected, degree

import scipy.sparse as ssp

import joblib  # Make ogb loads faster
from ogb.linkproppred import PygLinkPropPredDataset


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
METRIC_DIR = os.path.join(FILE_DIR, "..", "..", "data", "metrics")
PPR_DIR = os.path.join(FILE_DIR, "..", "..", "..", "lpformer", "node_subsets", "ppr")


def filter_by_year(data, split_edge, year=2007):
    """
    From BUDDY code

    remove edges before year from data and split edge
    @param data: pyg Data, pyg SplitEdge
    @param split_edges:
    @param year: int first year to use
    @return: pyg Data, pyg SplitEdge
    """
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge



def read_data_ogb(args):
    """
    Read data for OGB datasets
    """
    data_obj = {
        "dataset": args.dataset,
    }

    print("Loading all data...")

    dataset = PygLinkPropPredDataset(name=args.dataset)
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    if "collab" in args.dataset:
        data, split_edge = filter_by_year(data, split_edge)

    data_obj['num_nodes'] = data.num_nodes
    data_obj['edge_index'] = data.edge_index

    if args.dataset != 'ogbl-citation2':
        data_obj['test_pos'] = split_edge['test']['edge'].t()
    else:
        source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
        data_obj['test_pos'] = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1)

    # Add inverse so symmetric
    idx = torch.tensor([1,0])
    edge_index = torch.cat([data.edge_index, data.edge_index[idx]], dim=1)

    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        edge_weight = data.edge_weight.to(torch.float)
        edge_weight = torch.cat([edge_weight, edge_weight])
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    else:
        edge_weight = torch.ones(edge_index.size(1)).float().unsqueeze(-1)
            
    data_obj['adj'] = ssp.csr_matrix((edge_weight.squeeze(1), (edge_index[0], edge_index[1])), shape=(data.num_nodes, data.num_nodes))
 
    if args.use_val_in_test:
        val_edge_index = split_edge['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index)

        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data['full_edge_index'] = full_edge_index

        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
        val_edge_weight = torch.cat([edge_weight, val_edge_weight], 0).view(-1)
        data_obj['adj'] = ssp.csr_matrix((val_edge_weight, (full_edge_index[0], full_edge_index[1])), 
                                          shape=(data.num_nodes, data.num_nodes))

    data_obj['degree'] = degree(edge_index[0], num_nodes=data_obj['num_nodes'])
    if args.use_val_in_test:
        data_obj['degree'] = degree(full_edge_index[0], num_nodes=data_obj['num_nodes'])

    ### Load PPR matrix
    ### HACK: Stored as a SparseTensor. Convert to torch.sparse
    print("Reading PPR...", flush=True)
    ppr_dir = os.path.join(PPR_DIR, args.dataset)
    data_obj['ppr'] = torch.load(os.path.join(ppr_dir, f"sparse_adj-015_eps-{str(args.eps).replace('.', '')}.pt"))
    data_obj['ppr'] = data_obj['ppr'].to_torch_sparse_coo_tensor()

    if args.use_val_in_test:
        data_obj['ppr'] = torch.load(os.path.join(ppr_dir, f"sparse_adj-015_eps-{str(args.eps).replace('.', '')}_val.pt"))
        data_obj['ppr'] = data_obj['ppr'].to_torch_sparse_coo_tensor()

    return data_obj


def calc_degree(data, batch_size=10000):
    """
    Get scores of both nodes and combine via:
        - mean
        - min
        - max
    """
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(data['test_pos'].size(1)), batch_size)
    
    all_min, all_max, all_mean = [], [], []
    
    for ind in tqdm(link_loader, "Degree"):
        src, dst = data['test_pos'][0, ind], data['test_pos'][1, ind]
        src_degree = data['degree'][src]
        dst_degree = data['degree'][dst]

        min_degree = torch.where(src_degree > dst_degree, dst_degree, src_degree)
        max_degree = torch.where(src_degree > dst_degree, src_degree, dst_degree)
        mean_degree = (src_degree + dst_degree) / 2

        all_min.extend(min_degree.flatten().tolist())
        all_max.extend(max_degree.flatten().tolist())
        all_mean.extend(mean_degree.flatten().tolist())

    return all_min, all_max, all_mean


def calc_CN(data, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(data['test_pos'].size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader, "CNs"):
        src, dst = data['test_pos'][0, ind], data['test_pos'][1, ind]
        cur_scores = np.array(np.sum(data['adj'][src].multiply(data['adj'][dst]), 1)).flatten()
        scores.append(cur_scores)
        # print('max cn: ', np.concatenate(scores, 0).max())

    return np.concatenate(scores, 0)


def calc_shortest_path(A, edge_index, remove=True):
    
    scores = []
    G = nx.from_scipy_sparse_array(A)
    add_flag1 = 0
    add_flag2 = 0
    count = 0
    count1 = count2 = 0

    for i in tqdm(range(edge_index.size(1)), "SP"):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()
        if s == t:
            count += 1
            scores.append(999)
            continue

        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1

        if nx.has_path(G, source=s, target=t):
            sp = nx.shortest_path_length(G, source=s, target=t)
        else:
            sp = 999

        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
    
        scores.append(sp)

    return scores


def calc_ppr(data, batch_size=5000):
    """
    Get PPR scores for test samples

    """
    link_loader = DataLoader(range(data['test_pos'].size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader, "PPR"):
        src, dst = data['edge_index'][0, ind], data['edge_index'][1, ind]

        src_ppr = torch.index_select(data['ppr'], 0, src).to_dense()
        dst_ppr = torch.index_select(data['ppr'], 0, dst).to_dense()

        ppr_sd = src_ppr[torch.arange(len(dst)), dst]
        ppr_ds = dst_ppr[torch.arange(len(src)), src]
        ppr_vals = (ppr_sd + ppr_ds) / 2

        scores.extend(ppr_vals.tolist())

    return scores
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbl-collab")
    parser.add_argument("--eps", help="For PPR...", type=float, default=5e-5)
    args = parser.parse_args()

    args.use_val_in_test = False
    if "collab" in args.dataset:
        args.use_val_in_test = True
    
    data = read_data_ogb(args)

    if "ppa" in args.dataset.lower():
        data['test_pos'] = data['test_pos'][:, :200000]

    pos_test_edges = data['test_pos'].t().tolist()
    df = pd.DataFrame(pos_test_edges, columns=["src", "dst"])

    df['CN'] = calc_CN(data)
    df['SP'] = calc_shortest_path(data['adj'], data['test_pos'])
    df['PPR'] = calc_ppr(data)
    df['min_degree'], df['max_degree'], df['mean_degree'] = calc_degree(data)

    dir = os.path.join(METRIC_DIR, "ogb")
    os.makedirs(dir, exist_ok=True)
    df.to_csv(os.path.join(dir, f"{args.dataset}_metrics.csv"))


if __name__ == "__main__":
    main()