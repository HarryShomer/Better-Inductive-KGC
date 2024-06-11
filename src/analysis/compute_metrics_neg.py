import os
import torch
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from datetime import datetime 
import pickle
from collections import defaultdict
from torch_geometric.utils import degree, to_undirected
from time import perf_counter

from kgpy import datasets as kgpy_datasets
from util import *


def get_unq_rels(edge_index, edge_type, trips, num_ents):
    """
    Get "rel degree" for a node
    """
    edge_index = edge_index.t().tolist()
    edge_type = edge_type.tolist()

    ent_rels = [set() for _ in range(num_ents)]
    for ix, ei in enumerate(edge_index):
        ent_rels[ei[0]].add(edge_type[ix])

    rel_degree = []
    for t in trips:
        rel_degree.append(len(ent_rels[t[0]]))

    return rel_degree


def get_degree(edge_index, trips, num_nodes):
    """
    """
    node_degree = degree(edge_index[0], num_nodes)

    head_degree, tail_degree = [], []
    for t in trips:
        head_degree.append(node_degree[t[0]].item())
        tail_degree.append(node_degree[t[1]].item())

    return head_degree, tail_degree


def get_sp(edge_index, trips, num_nodes, split="test"):
    """
    """
    G = nx.Graph()
    G.add_nodes_from(list(range(num_nodes)))
    G.add_edges_from(edge_index.t().tolist())

    all_shortest_lengths = []

    for trip in tqdm(trips, "Calculating Shortest Path"):
        h, t = trip[0], trip[1]
        # Remove any edge connecting the 2 nodes
        # Just save val of elements and then add back in
        add_flag1, add_flag2 = 0, 0
        if split == "train":
            if (h, t) in G.edges: 
                G.remove_edge(h, t)
                add_flag1 = 1
            if (t, h) in G.edges: 
                G.remove_edge(t, h)
                add_flag2 = 1

            try:
                l = nx.shortest_path_length(G, source=h, target=t)
            except nx.exception.NetworkXNoPath:
                l = 100 

            if add_flag1 == 1: 
                G.add_edge(h, t)
            if add_flag2 == 1: 
                G.add_edge(t, h)
        else:
            try:
                l = nx.shortest_path_length(G, source=h, target=t)
            except nx.exception.NetworkXNoPath:
                l = 100 
        all_shortest_lengths.append(l)
    
    return all_shortest_lengths


def get_walks(edge_index, test_ents, num_nodes, length=6, prob=False):
    """
    """
    if prob:
        node_degree = degree(edge_index[0])
        edge_index_deg = torch.index_select(node_degree, 0, edge_index[0])
        edge_vals = 1 / edge_index_deg
    else:
        edge_vals = torch.ones(edge_index.size(1))

    A_sparse = torch.sparse_coo_tensor(edge_index, edge_vals, size=(num_nodes, num_nodes))

    cur_A = A_sparse.clone()
    all_walks = defaultdict(list)

    for l in range(1, length+1):
        print(datetime.now().strftime('%H:%M:%S'), ">>>", l)
        if l == 1:
            cur_A = A_sparse
        else:
            cur_A = cur_A @ A_sparse

        for t in tqdm(test_ents, "Assigning..."):
            all_walks[l].append(cur_A[t[0]][t[1]].item())
    
    all_walks = np.array([u for u in all_walks.values()]).T.tolist()

    return all_walks


def get_unique(edge_index, test_ents, num_nodes, length=6):
    """
    # of Unique nodes encountered on walks <= length 
    """
    edge_index = edge_index
    edge_vals = torch.ones(edge_index.size(1))
    A_sparse = torch.sparse_coo_tensor(edge_index, edge_vals, size=(num_nodes, num_nodes))

    BS = 10000
    node2unique = defaultdict(set)

    cur_A = A_sparse.clone()
    all_unique = defaultdict(list)

    for l in range(1, length+1):
        print(datetime.now().strftime('%H:%M:%S'), ">>>", l)
        if l == 1:
            cur_A = A_sparse
        else:
            cur_A = cur_A @ A_sparse

        # Track unique nodes
        for lower_ix in tqdm(range(0, num_nodes, BS), "Unique Nodes"):
            upper_ix = min(lower_ix + BS, num_nodes-1)
            bs_node_ix = torch.Tensor(list(range(lower_ix, upper_ix))).long()

            A_bs = torch.index_select(cur_A, 0, bs_node_ix).to_dense()
            A_nonzero = torch.nonzero(A_bs).tolist()

            for a in A_nonzero:
                # Account for index change in A_bs
                node2unique[a[0] + lower_ix].add(a[1])

        for t in tqdm(test_ents, "Assigning..."):
            all_unique[l].append(len(node2unique[t[0]]))
    
    all_unique = np.array([u for u in all_unique.values()]).T.tolist()

    return all_unique


def get_corresponding_pos_metrics(pos_neg_data, posdata, pos_samples):
    """
    For each negative sample, we want the metric for their positive sample

    1. PPR Range samples
    2. Prob Range samples
    """
    # Map entity pair to metrics 
    pos2data = {}
    for ix, ps in enumerate(pos_samples):
        pos2data[ps] = [posdata['sp'][ix], posdata['head_deg'][ix], posdata['tail_deg'][ix], 
                        posdata['ppr'][ix], posdata['prob'][ix]]

    ### PPR Range
    corresponding_src = [t[0] for t in pos_neg_data['neg_idx_samples']]
    corresponding_dst = pos_neg_data['corresponding_pos_tail']

    ppr_corresponding_pos_data = []
    for s, d in zip(corresponding_src, corresponding_dst):
        ppr_corresponding_pos_data.append(pos2data[(s, d)])
    
    ### Prob Range
    corresponding_src = [t[0] for t in pos_neg_data['neg_prob_range_ix']]
    corresponding_dst = pos_neg_data['neg_prob_range_pos_tail']

    prob_corresponding_pos_data = []
    for s, d in zip(corresponding_src, corresponding_dst):
        prob_corresponding_pos_data.append(pos2data[(s, d)])
    
    return ppr_corresponding_pos_data, prob_corresponding_pos_data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15K_237")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    is_valid = "_valid" if args.split == "valid" else ""

    samples_dataset = args.dataset
    if args.version is not None:
        samples_dataset = "ind" + args.dataset + args.version
    else:
        samples_dataset = samples_dataset.replace("_", "-")

    if args.version is not None:
        conf = {
            "name": args.dataset.replace("_", "-"),
            "root": "~/datasets/knowledge_graphs/",
            "version": args.version
        }
        dataset = IndRelLinkPredDataset(**conf)
        args.dataset = f"{args.dataset}_{args.version}"
        test_data = dataset[2] if args.split == "test" else dataset[1]
        edge_index = test_data.edge_index
        edge_type = test_data.edge_type
        num_nodes = test_data.num_nodes
        num_rels = dataset.num_relations // 2
    else:
        data = getattr(kgpy_datasets, args.dataset.upper().replace("-", "_"))(inverse=True)
        edge_index, edge_type = data.get_edge_tensors()
        num_nodes = data.num_entities
        num_rels = data.num_relations


    ### 1. Get data for positive samples
    pos_data = {}

    # [(h, r, t), ...]
    pos_samples = torch.load(os.path.join(PRED_DIR, f"nbfnet_{samples_dataset.lower()}_samples{is_valid}.pt"))
    if args.version is None:
        pos_samples = nbf_ids_to_ours_torch(pos_samples, data.dataset_name).long()
    pos_ents = [(t[0], t[2]) for t in pos_samples.t().tolist()]
    
    pos_data['head_deg'], pos_data['tail_deg'] = get_degree(edge_index, pos_ents, num_nodes)
    pos_data['sp'] = get_sp(edge_index, pos_ents, num_nodes, "test")
    pos_data['rel_deg'] = get_unq_rels(edge_index, edge_type, pos_ents, num_nodes)
    # pos_data['walks'] = get_walks(edge_index, pos_ents, num_nodes)
    # pos_data['unq'] = get_unique(edge_index, pos_ents, num_nodes)


    ### 2. Get data for negative samples
    neg_data = {"prob": {}, "ppr": {}}
    with open(os.path.join(METRIC_DIR, f'{args.dataset}_probs_by_ppr_range{is_valid}.pkl'), 'rb') as handle:
        pos_neg_data = pickle.load(handle)

    pos_data['ppr'] = pos_neg_data['all_pos_pprs']
    pos_data['prob'] = pos_neg_data['all_pos_probs']
    
    ### PPR Negs
    neg_samples = pos_neg_data['neg_idx_samples']
    neg_data['ppr']['head_deg'], neg_data['ppr']['tail_deg'] = get_degree(edge_index, neg_samples, num_nodes)
    neg_data['ppr']['sp'] = get_sp(edge_index, neg_samples, num_nodes, "test")
    neg_data['ppr']['rel_deg'] = get_unq_rels(edge_index, edge_type, neg_samples, num_nodes)
    # neg_data['walks'] = get_walks(edge_index, neg_samples, num_nodes)
    # neg_data['unq'] = get_unique(edge_index, neg_samples, num_nodes)
    neg_data['ppr']['ppr'] = pos_neg_data['neg_ppr_samples']
    neg_data['ppr']['prob'] = pos_neg_data['neg_pred_samples']

    ### Prob Negs
    neg_samples = pos_neg_data['neg_prob_range_ix']
    neg_data['prob']['head_deg'], neg_data['prob']['tail_deg'] = get_degree(edge_index, neg_samples, num_nodes)
    neg_data['prob']['sp'] = get_sp(edge_index, neg_samples, num_nodes, "test")
    neg_data['prob']['rel_deg'] = get_unq_rels(edge_index, edge_type, neg_samples, num_nodes)
    # neg_data['walks'] = get_walks(edge_index, neg_samples, num_nodes)
    # neg_data['unq'] = get_unique(edge_index, neg_samples, num_nodes)
    neg_data['prob']['ppr'] = pos_neg_data['neg_prob_range_ppr']
    neg_data['prob']['prob'] = pos_neg_data['neg_prob_range_prob']

    a, b = get_corresponding_pos_metrics(pos_neg_data, pos_data, pos_ents)
    neg_data['ppr']['pos_metrics'], neg_data['prob']['pos_metrics'] = a, b
    final_data = {"pos": pos_data, "neg": neg_data, "num_neg_in_prob_range": pos_neg_data['num_neg_by_prob_range']}

    with open(os.path.join(METRIC_DIR, f"{args.dataset}_pos_neg_metrics{is_valid}.pkl"), "wb") as f:
        pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()