import os
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.utils import degree

from kgpy import datasets as kgpy_datasets
from util import *

import sys
sys.path.append("..")
import calc_ppr_matrices as calcppr

NEW_PPR_DIR = os.path.join(FILE_DIR, "..", "..", "data", "modify_ppr")


def save_results(edges2add, ent2trips, dataset, trip_type="low"):
    ddir = os.path.join(NEW_PPR_DIR, dataset)
    if not os.path.isdir(os.path.dirname(ddir)):
        os.makedirs(os.path.dirname(ddir), exist_ok=True)

    with open(os.path.join(ddir, f"ent2trips_{trip_type}.pkl"), "wb") as f:
        pickle.dump(ent2trips, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(ddir, f"edges2add_{trip_type}.pkl"), "wb") as f:
        pickle.dump(edges2add, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_ppr_new_graph(trips, edge_index, num_ents):
    """
    Calc PPR triples for the new graph with added edges
    """
    neighbors, neighbor_weights = calcppr.get_ppr_matrix(edge_index, num_ents, 0.15, 1e-5, verbose=False)
    ppr_matrix = calcppr.create_sparse_ppr_matrix(neighbors, neighbor_weights)

    new_ppr = []
    for t in trips:
        new_ppr.append(ppr_matrix[t[0]][t[1]].item())
        
    return new_ppr



def group_samples_by_entity(trips, trips_ppr, group_by="tail"):
    """
    group_by = ['head', 'tail']
    """
    ent_ix = 2 if group_by == "tail" else 0
    other_ent_ix = 0 if group_by == "tail" else 2

    # Elements in list structured as (rel, other entity, ppr)
    ent2trips = defaultdict(list)

    for t, p in zip(trips, trips_ppr):
        ent2trips[t[ent_ix]].append([t[1], t[other_ent_ix], p])
    
    return ent2trips



def get_ppr_samples(args):
    """
    Read the sparse PPR matrix
    """
    # root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
    # d = args.dataset_name.upper() if args.version is None else args.dataset_name
    # dataset_dir = os.path.join(root_dir, "ppr", d)
    # ppr = torch.load(os.path.join(dataset_dir, f"sparse_adj-015_eps-{args.eps}".replace(".", "") + ".pt"))

    ppr_df = pd.read_csv(os.path.join(PPR_DIR, args.dataset_name, f"test_ppr_eps-{str(args.eps).replace('.', '')}.csv")) 
    ppr_df = ppr_df.sort_values(by=['head', 'rel', 'tail'])
    all_samples = ppr_df[['head', 'rel', 'tail']].to_numpy()
    ppr_vals = ppr_df['ppr'].to_numpy()

    # PPR with low values (< 1e-4)
    low_ppr_samples = all_samples[ppr_vals < 1e-4]
    low_ppr_vals = ppr_vals[ppr_vals < 1e-4]
    # PPR with high values (> 1e-2)
    high_ppr_samples = all_samples[ppr_vals > 1e-2]
    high_ppr_vals = ppr_vals[ppr_vals > 1e-2]

    print("\n# Samples with Low PPR:", len(low_ppr_samples))
    print("# Samples with High PPR:", len(high_ppr_samples))

    ent2trips_low = group_samples_by_entity(low_ppr_samples, low_ppr_vals)
    ent2trips_high = group_samples_by_entity(high_ppr_samples, high_ppr_vals, group_by="head")

    return ent2trips_low, ent2trips_high



def add_edges_for_samples(edge_index, num_ents, ent2trips, num_edges, max_trips=250):
    """
    1. Add `num_edges` new edges for each
    2. Calc new PPR for existing triples
    3. Stop when we covered enough samples! 
    """
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), size=(num_ents, num_ents))
    
    # Map entity to list of new edges of form
    # Just store entity being connected to
    edges2add = {}
    num_trips_so_far = 0

    for ent in ent2trips:
        ent_row = adj[ent].to_dense().bool()
        ent_no_edge = (ent_row == 0).nonzero()

        # Randomly select `num_edges` w/o replacement
        idx = torch.randperm(len(ent_no_edge))[:num_edges]
        new_edges = ent_no_edge[idx].flatten()
        edges2add[ent] = new_edges.tolist()

        # Add new edges ...
        new_edge_torch = torch.Tensor([(ent, nn.item()) for nn in new_edges])
        new_edge_torch_inv = torch.Tensor([(nn.item(), ent) for nn in new_edges])
        new_edge_index = torch.cat((edge_index.t(), new_edge_torch, new_edge_torch_inv)).t()

        # Calc ppr on new graph
        ent_trips = [(ent, eee[1]) for eee in ent2trips[ent]]
        new_ppr = get_ppr_new_graph(ent_trips, new_edge_index.long(), num_ents)

        old_ppr = [eee[-1] for eee in ent2trips[ent]]
        mean_change_ppr = np.mean(np.array(new_ppr) - np.array(old_ppr))
        print("\nMean Change in PPR:", round(mean_change_ppr, 6))

        # Add new ppr score to ent2trips as last entry in each row
        for ix in range(len(ent2trips[ent])):
            ent2trips[ent][ix].append(new_ppr[ix])

        num_trips_so_far += len(ent2trips[ent])
        print("# Trips =", num_trips_so_far)

        if num_trips_so_far >= max_trips:
            break
    
    return edges2add, ent2trips



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15K_237")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--num-edges", help="Number of edges to add", type=int, default=20)
    parser.add_argument("--max-trips", help="Number of triples to consider", type=int, default=500)
    args = parser.parse_args()

    if args.version is None:
        data = getattr(kgpy_datasets, args.dataset.upper())(inverse=True)
        args.eps = 1e-6
        args.dataset_name = args.dataset  #.replace("-", "_")
        edge_index = data.get_edge_tensors()
        num_ents = data.num_entities

        valid_ents = torch.Tensor([(t[0], t[2]) for t in data['valid']]).long()
        test_ents = torch.Tensor([(t[0], t[2]) for t in data['test']]).long()
        all_edge_index = torch.cat((edge_index.t(), valid_ents, test_ents)).t()
    else:
        conf = {
            "name": args.dataset.replace("_", "-"),
            "root": "~/datasets/knowledge_graphs/",
            "version": args.version
        }
        args.eps = 1e-7
        args.dataset_name = f"{args.dataset}_{args.version}"
        dataset = IndRelLinkPredDataset(**conf)
        test_data = dataset[2]
        edge_index = test_data.edge_index
        num_ents = test_data.num_nodes

        # Add test edges to propagation graph so we don't accidentally sample them which can inflate performance
        all_edge_index = torch.cat((edge_index.t(), test_data.target_edge_index.t())).t()

    ent2trips_low, ent2trips_high = get_ppr_samples(args)

    edges2add_low, ent2trips_low = add_edges_for_samples(all_edge_index, num_ents, ent2trips_low, 
                                                         args.num_edges, max_trips=args.max_trips)
    save_results(edges2add_low, ent2trips_low, args.dataset_name, trip_type="low")

    edges2add_high, ent2trips_high = add_edges_for_samples(all_edge_index, num_ents, ent2trips_high, 
                                                           args.num_edges, max_trips=args.max_trips)
    save_results(edges2add_high, ent2trips_high, args.dataset_name, trip_type="high")



if __name__ == "__main__":
    main()