import os
import torch
import random
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree

from utils import *
from calc_ppr_matrices import get_ppr_matrix, create_sparse_ppr_matrix, save_results


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
METRIC_DIR = os.path.join(FILE_DIR, "..", "..", "data", "metrics")


# def get_ppr(cfg, data):
#     """
#     Get PPR matrix.

#     If it exists, load it in

#     Otherwise create and save it
#     """
#     print("Loading PPR...")
#     ddir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "ppr", cfg['dataset_name'])
#     file_name = f"sparse_adj-{cfg['alpha']}_eps-{cfg['eps']}".replace(".", "") + ".pt"
    
#     if os.path.isfile(os.path.join(ddir, file_name)):
#         ppr_matrix = torch.load(os.path.join(ddir, file_name))
#     else:
#         alpha, eps = cfg['alpha'], cfg['eps'] 
#         neighbors, neighbor_weights = get_ppr_matrix(data.edge_index, data.num_nodes, alpha, eps)
#         ppr_matrix = create_sparse_ppr_matrix(neighbors, neighbor_weights)
#         save_results(cfg['dataset_name'], ppr_matrix, alpha, eps, val = False)
    
#     return ppr_matrix



def create_nx_graph(data, num_rels):
    """
    1. Ignore relations
    2. Only allow 1 edge between nodes
    3. Weight of edge = # edges/relations connecting nodes
    """
    print("Constructing NetworkX Graph...")

    # Remove reciprocal relations
    non_inv_edges = data.edge_type < num_rels
    edge_index = data.edge_index[:, non_inv_edges].t().tolist()

    unique_ents = set()
    for e in edge_index:
        unique_ents.add(e[0])
        unique_ents.add(e[1])
    
    edge2numoccur = defaultdict(int)
    for e in edge_index:
        edge2numoccur[tuple(e)] += 1
    
    edge_plus_weight = []
    for e, v in edge2numoccur.items():
        edge_plus_weight.append((e[0], e[1], v))

    G = nx.Graph()
    G.add_weighted_edges_from(edge_plus_weight)

    return G, unique_ents


def load_ppr(args):
    """
    Load sparse ppr matrix
    """
    dataset_name = args.dataset 
    if args.version is not None:
        dataset_name += f"_{args.version}"

    ddir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "ppr", dataset_name)
    file_name = f"sparse_adj-{args.alpha}_eps-{args.eps}".replace(".", "")  + ".pt"
    ppr_matrix = torch.load(os.path.join(ddir, file_name))

    return ppr_matrix


def sp2bin(sp_val, sp_bins):
    """
    Get corresonding bin for sp value
    """
    for sp_b in sp_bins:
        if sp_val >= sp_b[0] and sp_val < sp_b[1]:
            return sp_b
    
    return sp_bins[-1]


def detect_communities(args, data, num_rels):
    """
    1. Louvain
    2. LabelProp
    """
    G, _ = create_nx_graph(data, num_rels)

    if args.alg.lower() == "louvain":
        comm = nx.community.louvain_communities(G, weight="weight", seed=42)
    else:
        comm = nx.community.asyn_lpa_communities(G, weight="weight", seed=42)
        comm = [c for c in comm]

    # print(f"\n# Normalized Communities = {len(comm) / data.num_nodes}")

    return comm


def get_ppr_by_sample(data, args, neg_samples):
    """
    Get for both for (a, b) and (b, a).

    Also generate some negative and get their PPR as well

    Returns 
        samples, ppr for sample 
    """
    all_ppr_tgt, all_ppr_tgt_inv = [], []

    test_ents = data.target_edge_index.t()
    link_loader = DataLoader(range(len(test_ents)), 10000)

    ppr_matrix = load_ppr(args)

    for ind in tqdm(link_loader, "Pos PPR"):
        src, dst = test_ents[ind, 0].long(), test_ents[ind, 1].long()

        ### DST PPR
        dst_ppr = torch.index_select(ppr_matrix, 0, dst).to_dense()
        ppr_ds = dst_ppr[torch.arange(len(dst)), src]
        all_ppr_tgt.extend(ppr_ds.tolist())

        # We need to include the inverse since the Data object doesn't include it
        src_ppr = torch.index_select(ppr_matrix, 0, src).to_dense()
        ppr_sd = src_ppr[torch.arange(len(src)), dst]
        all_ppr_tgt_inv.extend(ppr_sd.tolist())
    
    all_pos_ppr = all_ppr_tgt + all_ppr_tgt_inv
    all_pos_samples = torch.cat((data.target_edge_index, data.target_edge_index[[1, 0]]), dim=-1)

    ### Negative data
    # Generate (num_pos * k) negatives randomly
    neg_samples = torch.Tensor(neg_samples).long()
    link_loader = DataLoader(range(len(neg_samples)), 10000)

    all_neg_ppr = []
    for ind in tqdm(link_loader, "Neg PPR"):
        src, dst = neg_samples[ind, 0].long(), neg_samples[ind, 1].long()

        dst_ppr = torch.index_select(ppr_matrix, 0, dst).to_dense()
        ppr_ds = dst_ppr[torch.arange(len(dst)), src]
        all_neg_ppr.extend(ppr_ds.tolist())

    return all_pos_samples.t().tolist(), all_pos_ppr, all_neg_ppr




def ppr_by_sp(data, args):
    """
    Get the mean PPR for links in same/different communities

    Also break down by shortest path length bec. why not
    """        
    neg_samples = gen_k_negs_per_pos(data, args.num_negs)
    # pos_samples, pos_ppr, neg_ppr = get_ppr_by_sample(data, args, neg_samples)
    pos_samples = torch.cat((data.target_edge_index, data.target_edge_index[[1, 0]]), dim=-1).t().tolist()
    
    pos_sp = calc_path_length(data, pos_samples)
    neg_sp = calc_path_length(data, neg_samples)
    pos_sp, neg_sp = np.array(pos_sp), np.array(neg_sp)

    # sp_bins = [(1, 2), (2, 3), (3, 4), (4, 101)]

    # ### Positive Samples ###
    # pos_ppr_by_sp = {s : [] for s in sp_bins}
    # for e, e_ppr, e_sp in zip(pos_samples, pos_ppr, pos_sp):
    #     e_sp_bin = sp2bin(e_sp, sp_bins)
    #     pos_ppr_by_sp[e_sp_bin].append(e_ppr)

    # ### Positive Samples ###
    # neg_ppr_by_sp = {s : [] for s in sp_bins}
    # for e, e_ppr, e_sp in zip(neg_samples, neg_ppr, neg_sp):
    #     e_sp_bin = sp2bin(e_sp, sp_bins)
    #     neg_ppr_by_sp[e_sp_bin].append(e_ppr)

    # print(f"\n>>> PPR By SP")
    # for sp_b in sp_bins:
    #     print(f"\n{sp_b}:\n--------")
    #     print(f"Mean Positive PPR (# {len(pos_ppr_by_sp[sp_b])}) = {np.mean(pos_ppr_by_sp[sp_b])}")
    #     print(f"Mean Negative PPR (# {len(neg_ppr_by_sp[sp_b])}) = {np.mean(neg_ppr_by_sp[sp_b])}")
    

    # max_sp = 10
    # pos_sp[pos_sp >= max_sp] = max_sp
    # neg_sp[neg_sp >= max_sp] = max_sp

    # print(f"\nMean Pos/Neg SP = {np.mean(pos_sp)} / {np.mean(neg_sp)}")

    # print(f"Mean Connected Pos/Neg SP = {np.mean(pos_sp[pos_sp < 100])} / {np.mean(neg_sp[neg_sp < 100])}")
    # print(f"% Pos/Neg Disconnected = {(pos_sp == 100).sum() / len(pos_sp)} / {(neg_sp == 100).sum() / len(neg_sp)}")    
    print(f"SP EMD = {calc_emd_dist(pos_sp, neg_sp)}")


    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15K_237")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--alg", help="louvain or labelprop", type=str, default="louvain")
    parser.add_argument("--num-negs", help="Negs per positive", type=int, default=25)
    parser.add_argument("--eps", help="Stopping criterion threshold", type=float, default=1e-7)
    parser.add_argument("--alpha", help="Teleportation probability", type=float, default=0.15)

    parser.add_argument("--num-test", help="# Test Graphs", type=int, default=1)
    parser.add_argument("--new", help="New Splits", action="store_true", default=False)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    dataset = build_dataset(args.dataset, args.version, args.num_test, args.new)

    if args.new:
        for i in range(args.num_test):
            print(f"TEST GRAPH {i}:\n------------")
            ppr_by_sp(dataset[i+2], args)
    else:
        ppr_by_sp(dataset[2], args)

    exit()

    ddir = dataset.raw_dir
    train_trips = read_data(os.path.join(ddir, "train_graph.txt"))
    train_rels = [t[1] for t in train_trips]
    train_rels = set(train_rels)

    # # \% of Triples with new relations
    for i in range(args.num_test):
        print(f"\n>>> Inference Graph {i}")

        inf_graph_trips = read_data(os.path.join(ddir, f"test_{i}_graph.txt"))
        inf_graph_rels = [t[1] for t in inf_graph_trips]

        inf_test_trips = read_data(os.path.join(ddir, f"test_{i}_samples.txt"))
        inf_test_rels = [t[1] for t in inf_test_trips]

        num_new_rels = 0
        for r in inf_graph_rels:
            if r not in train_rels:
                num_new_rels += 1 
        print(f"  % New Rels in Graph: {num_new_rels/len(inf_graph_rels):.4f}")

        num_new_rels = 0
        for r in inf_test_rels:
            if r not in train_rels:
                num_new_rels += 1 
        print(f"  # New/Existing Rels in Test:", num_new_rels, len(inf_test_rels) - num_new_rels)
        print(f"  % New Rels in Test: {num_new_rels/len(inf_test_rels):.4f}")



if __name__ == "__main__":
    main()