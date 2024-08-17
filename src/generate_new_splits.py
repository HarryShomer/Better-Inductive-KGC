"""
Generate new data splits using clustering
"""
import os
import torch
import random
import argparse
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import torch_geometric.transforms as T
from sklearn.cluster import SpectralClustering
from torch_geometric.utils import  to_undirected, remove_self_loops, coalesce, to_scipy_sparse_matrix

import utils
from calc_ppr_matrices import *

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "new_data")


def save_data(args, data_list, filenames):
    """
    corresponding entries in data_list in filenames are their name
    """
    if args.save_as is None:
        raise ValueError("Must pass name via --save-as args to save final data")
    
    ddir = os.path.join(DATA_DIR, args.save_as)
    os.makedirs(ddir, exist_ok=True)

    print("\nSaving Data...")
    for dobj, fname in zip(data_list, filenames):
        with open(os.path.join(ddir, fname), "w") as f:
            for e in dobj:
                f.write(f"{e[0]} {e[1]} {e[2]}\n")



def get_edge_weight(data):
    """
    Equal to # of edges btwn node pair
    """    
    edges = data.edge_index.t().tolist()

    edge2numoccur = defaultdict(int)
    for e in edges:
        edge2numoccur[tuple(e)] += 1
    
    edge_weight = []
    for e in edges:
        edge_weight.append(edge2numoccur[tuple(e)])
    
    return torch.Tensor(edge_weight)


def get_lcc(edges, nodes):
    """
    Largest connected components

    Need to be cognizant of relations
    """
    G = nx.Graph()
    G.add_weighted_edges_from([(e[0], e[2], e[1]) for e in edges])
    if nx.number_connected_components(G) == 1:
        return edges, nodes

    print("Getting LCC...")
    print(f"# Edges B4: {len(edges)}")

    largest_cc = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc = G.subgraph(largest_cc[0])

    new_nodes = set(list(lcc.nodes))
    new_edges = []

    for e in edges:
        if e[0] in new_nodes or e[2] in new_nodes:
            new_edges.append(e)
    
    print(f"# Edges After: {len(new_edges)}")

    return new_edges, new_nodes


def get_perc_new_rels(rels1, rels2):
    rels1_unq, rels2_unq = set(rels1), set(rels2)

    num_diff = 0
    for r in rels1:
        if r not in rels2_unq:
            num_diff += 1 
            
    return num_diff / len(rels1)


def data_preprocessing_louvain(dataset):
    """
    Read data and:
        1. Combine train+valid+test
        2. Remove reciprocal edges
        3. Get LCC
        4. Get edge weight == # of relations btwn nodes
        5. Coalesce after creating edge weight

    Return data object with all data
    """
    num_rels = dataset.num_relations // 2
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]

    # Add in to val+test
    val_edges = torch.cat((valid_data.target_edge_index, valid_data.target_edge_index[[1, 0]]), dim=-1)
    test_edges = torch.cat((test_data.target_edge_index, test_data.target_edge_index[[1, 0]]), dim=-1)
    val_edge_types = torch.cat((valid_data.target_edge_type, valid_data.target_edge_type + num_rels))
    test_edge_types = torch.cat((test_data.target_edge_type, test_data.target_edge_type + num_rels))

    all_ei = torch.cat([train_data.edge_index, val_edges, test_edges], dim=-1)
    all_et = torch.cat([train_data.edge_type, val_edge_types, test_edge_types])

    # Remove reciprocal
    non_inv_edges = all_et < num_rels
    ei_noninv, et_noninv = all_ei[:, non_inv_edges], all_et[non_inv_edges]

    train_data.edge_index, train_data.edge_type = ei_noninv, et_noninv 
    train_data = T.Compose([T.LargestConnectedComponents()])(train_data)

    ei, _ = remove_self_loops(train_data.edge_index)
    ei = to_undirected(ei)

    edge2numoccur = defaultdict(int)
    for e in ei.t().tolist():
        edge2numoccur[tuple(e)] += 1
    
    edge_plus_weight = []
    for e, v in edge2numoccur.items():
        edge_plus_weight.append((e[0], e[1], v))

    G = nx.Graph()
    G.add_weighted_edges_from(edge_plus_weight)

    return train_data, G




def data_preprocessing_spectral(dataset):
    """
    Read data and:
        1. Combine train+valid+test
        2. Remove reciprocal edges
        3. Get LCC
        4. Get edge weight == # of relations btwn nodes
        5. Coalesce after creating edge weight

    Return data object with all data
    """
    num_rels = dataset.num_relations // 2
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]

    # Add in to val+test
    val_edges = torch.cat((valid_data.target_edge_index, valid_data.target_edge_index[[1, 0]]), dim=-1)
    test_edges = torch.cat((test_data.target_edge_index, test_data.target_edge_index[[1, 0]]), dim=-1)
    val_edge_types = torch.cat((valid_data.target_edge_type, valid_data.target_edge_type + num_rels))
    test_edge_types = torch.cat((test_data.target_edge_type, test_data.target_edge_type + num_rels))

    all_ei = torch.cat([train_data.edge_index, val_edges, test_edges], dim=-1)
    all_et = torch.cat([train_data.edge_type, val_edge_types, test_edge_types])

    # Remove reciprocal
    non_inv_edges = all_et < num_rels
    ei_noninv, et_noninv = all_ei[:, non_inv_edges], all_et[non_inv_edges]

    train_data.edge_index, train_data.edge_type = ei_noninv, et_noninv 
    train_data = T.Compose([T.LargestConnectedComponents()])(train_data)

    train_data.edge_weight = get_edge_weight(train_data)

    ei_c, ew_c = coalesce(train_data.edge_index, train_data.edge_weight)
    train_data.coalesced_edge_index = ei_c 
    train_data.coalesced_edge_weight = ew_c

    return train_data
    

def cluster_spectral(data, args):
    """
    """
    ei, ew = remove_self_loops(data.coalesced_edge_index, data.coalesced_edge_weight)
    ei, ew = to_undirected(ei, ew)

    num_clusters = args.num_clusters
    A = to_scipy_sparse_matrix(ei, ew)

    print("Clustering data...")
    sc = SpectralClustering(n_clusters=num_clusters, eigen_solver="amg", affinity='precomputed',
                            assign_labels=args.assign, random_state=args.seed, n_jobs=10)
    node_clusters = torch.Tensor(sc.fit(A).labels_)

    cluster_nodes = defaultdict(list)
    for n in range(num_clusters):
        cluster_nodes[n] = set(torch.nonzero(node_clusters == n).flatten().tolist())
    
    return cluster_nodes


def cluster_data(args, data):
    """
    Cluster with either spectral or louvain

    Check diff clusters and compare them

    Should have reduced data to LCC already
    NOTE: It may still throw an error that it isn't "fully-connected" but it can be ignored
    """
    if args.alg == "spectral":
        num_clusters = args.num_clusters
        data = data_preprocessing_spectral(data)
        cluster_nodes = cluster_spectral(data, args)
    else:
        data, G = data_preprocessing_louvain(data)
        comm = nx.community.louvain_communities(G, weight="weight", seed=1)
        num_clusters = len(comm)

        cluster_nodes = defaultdict(list)
        for n, cx in enumerate(comm):
            cluster_nodes[n] = set(list(cx))

    cluster_edges = defaultdict(list)
    cluster_rels = defaultdict(set)

    original_ei = data.edge_index.t().tolist()
    original_et = data.edge_type.tolist()

    num_nodes = [] # Entry i == cluster i

    for e, r in zip(original_ei, original_et):
        for n in range(num_clusters):
            if e[0] in cluster_nodes[n] and e[1] in cluster_nodes[n]:
                cluster_edges[n].append((e[0], r, e[1]))
                cluster_rels[n].add(r)

    if args.print_candidates:
        print("\nResults:\n-------")
        for n in range(num_clusters):
            print(f"G{n}: |V|={len(cluster_nodes[n])}, |E|={len(cluster_edges[n])}, |R|={len(cluster_rels[n])}, "\
                f"d={round(len(cluster_edges[n]) / len(cluster_nodes[n]), 1)}")
            num_nodes.append(len(cluster_nodes[n]))

        topk_clusters = torch.topk(torch.Tensor(num_nodes), args.candidates).indices.tolist()
        print("\n>>> Overlap Between Different Clusters")
        for n1 in topk_clusters:
            for n2 in topk_clusters:
                if n1 != n2:
                    ents1 = cluster_nodes[n1]
                    ents2 = cluster_nodes[n2]
                    rels1 = cluster_rels[n1]
                    rels2 = cluster_rels[n2]
                    print("---------------------------------------")
                    print("G1/G2:", len(ents1), len(ents2), len(rels1), len(rels2))
 
                    rels1 = [t[1] for t in cluster_edges[n1]]
                    rels2 = [t[1] for t in cluster_edges[n2]]
                    print("# Rel1 \ Rel2:", f"({round(get_perc_new_rels(rels1, rels2), 2)})")
                    print("# Rel2 \ Rel1:", f"({round(get_perc_new_rels(rels2, rels1), 2)})")

        exit()  # Easier this way...


    if len(args.choose_graphs) > 0:
        gtrain = int(args.choose_graphs[0])

        if args.lcc:
            train_edges, train_nodes = get_lcc(cluster_edges[gtrain], cluster_nodes[gtrain])
        else:
            train_edges = cluster_edges[gtrain]
            train_nodes = cluster_nodes[gtrain]
        
        all_data = {
            "train_edges": train_edges,
            "train_nodes": train_nodes,
            "train_rels": cluster_rels[gtrain],
        }

        for ix, inf_i in enumerate(args.choose_graphs[1:]):
            inf_i = int(inf_i)
            inf_ents, inf_rels, inf_edges = cluster_nodes[inf_i], cluster_rels[inf_i], cluster_edges[inf_i]
            
            # Remove all unique relations in G_inf and their corresponding edges
            if args.type == "E":
                unique_inf_rels = inf_rels - cluster_rels[gtrain]
                inf_rels = inf_rels - unique_inf_rels

                filtered_inf_edges, filtered_inf_ents = [], set()
                for e in inf_edges:
                    if e[1] in inf_rels:
                        filtered_inf_ents.add(e[0])
                        filtered_inf_ents.add(e[2])
                        filtered_inf_edges.append(e)
                
                inf_ents = filtered_inf_ents
                inf_edges = filtered_inf_edges 

            if args.lcc:
                all_data[f'test_edges_{ix}'], _ = get_lcc(inf_edges, inf_ents)
            else:
                all_data[f'test_edges_{ix}'] = inf_edges

        return data, all_data



def get_new_data(args):
    """
    For (E) and (E, R) setting
    """
    fdata = {}

    data = utils.build_dataset(args.dataset)
    data, cldata = cluster_data(args, data)

    # Remove % of edges for valid graph
    random.shuffle(cldata['train_edges'])
    val_thresh = int(len(cldata['train_edges']) * (1-args.perc_holdout))
    fdata['train_graph'], fdata['valid_samples'] = cldata['train_edges'][:val_thresh], cldata['train_edges'][val_thresh:]

    if args.lcc:
        fdata['train_graph'], _ = get_lcc(fdata['train_graph'], [])

    # Calc nodes/rels in valid graph after spliiting. All valid sample must contain them
    filt_tr_nodes, filt_tr_rels = set(), set()
    for e in fdata[f'train_graph']:
        filt_tr_rels.add(e[1])
        filt_tr_nodes.add(e[0]) ; filt_tr_nodes.add(e[2])

    # Remove those that aren't in graph after spliiting from samples
    # Overwrite corresponding entry in fdata
    filt_val_samples = []
    for e in fdata['valid_samples']:
        if e[0] in filt_tr_nodes and e[1] in filt_tr_rels and e[2] in filt_tr_nodes:
            filt_val_samples.append(e)
    fdata['valid_samples'] = filt_val_samples

    # Remove % of edges for *each* test graph
    for ix in range(len(args.choose_graphs) - 1):
        random.shuffle(cldata[f'test_edges_{ix}'])
        inf_thresh = int(len(cldata[f'test_edges_{ix}']) * (1-args.perc_holdout))
        fdata[f'test_{ix}_graph'] = cldata[f'test_edges_{ix}'][:inf_thresh]
        fdata[f'test_{ix}_samples'] = cldata[f'test_edges_{ix}'][inf_thresh:]

        if args.lcc:
            fdata[f'test_{ix}_graph'], _ = get_lcc(fdata[f'test_{ix}_graph'], [])

        # Existing nodes/rels in test ix graph
        filt_test_nodes, filt_test_rels = set(), set()
        for e in fdata[f'test_{ix}_graph']:
            filt_test_rels.add(e[1])
            filt_test_nodes.add(e[0]) ; filt_test_nodes.add(e[2])

        filt_test_samples = []
        for e in fdata[f'test_{ix}_samples']:
            if e[0] in filt_test_nodes and e[1] in filt_test_rels and e[2] in filt_test_nodes:
                filt_test_samples.append(e)
        
        fdata[f'test_{ix}_samples'] = filt_test_samples
        fdata[f'test_{ix}_nodes'] = filt_test_nodes
        fdata[f'test_{ix}_rels'] = filt_test_rels


    ### Convert IDs to original names b4 saving
    ent2name, rel2name = data.entid_2_name, data.relid_2_name
    new_edges = []
    for e in fdata[f'train_graph']:
        new_edges.append((ent2name[e[0]], rel2name[e[1]], ent2name[e[2]]))
    
    # Overwrite fdata
    fdata[f'train_graph'] = new_edges
    fdata['valid_samples'] = [(ent2name[e[0]], rel2name[e[1]], ent2name[e[2]]) for e in fdata['valid_samples']]

    # Same for each test split
    for split in range(len(args.choose_graphs) - 1):
        for trip_type in ['graph', 'samples']:
            new_inf_edges = []

            for e in fdata[f'test_{split}_{trip_type}']:
                # print(split, trip_type, "==>", e)
                new_inf_edges.append((ent2name[e[0]], rel2name[e[1]], ent2name[e[2]]))
            
            fdata[f'test_{split}_{trip_type}'] = new_inf_edges

    
    print("\nGraph Sizes:\n-----------")
    tr_deg = round(len(fdata['train_graph']) / len(cldata['train_nodes']), 1)
    print(f"  G_train = {len(cldata['train_nodes'])}, {len(cldata['train_rels'])}, {len(fdata['train_graph'])}, {tr_deg}")
    print(f"  # Valid = {len(fdata['valid_samples'])}\n")

    for split in range(len(args.choose_graphs)-1):
        num_nodes = len(fdata[f'test_{split}_nodes'])
        num_rels = len(fdata[f'test_{split}_rels'])
        num_samples = len(fdata[f'test_{split}_samples'])
        num_edges = len(fdata[f'test_{split}_graph'])
        mean_degree = num_edges / num_nodes

        rels1 = [t[1] for t in fdata['train_graph']]
        rels2 = [t[1] for t in fdata[f'test_{split}_graph']]
        pnew_rels = get_perc_new_rels(rels2, rels1)
        print(f"  G_test {split} = {num_nodes}, {num_rels} ({pnew_rels}), {num_edges}, {num_samples}, {round(mean_degree, 1)}")

    if args.save_as is None:
        raise ValueError("Must pass name via --save-as args to save final data")
 
    ddir = os.path.join(DATA_DIR, args.save_as)
    os.makedirs(ddir, exist_ok=True)

    keys2save = ["train_graph", "valid_samples"]    
    keys2save.extend([f'test_{split}_graph' for split in range(len(args.choose_graphs) - 1)])
    keys2save.extend([f'test_{split}_samples' for split in range(len(args.choose_graphs) - 1)])

    print("\nSaving Data...")
    for fname in keys2save:
        with open(os.path.join(ddir, fname+".txt"), "w") as f:
            for e in fdata[fname]:
                f.write(f"{e[0]} {e[1]} {e[2]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="FB15k-237")
    parser.add_argument('--alg', type=str, default="spectral")
    parser.add_argument("--num-clusters", help="# of Clusters to Find", type=int, default=5)
    parser.add_argument('--type', help="(E) or (E, R) Inductive. Choices are 'E' or 'ER'", type=str, default="E")
    parser.add_argument("--perc-holdout", help="% of train/test graph to holdout for val/test", type=float, default=0.1)
    parser.add_argument('--assign', help="How to assign labels in SC", type=str, default="cluster_qr")

    parser.add_argument("--print-candidates", help="Print Candidate Clusters pairs", action="store_true", default=False)
    parser.add_argument("--candidates", help="# of Candidates to consider", type=int, default=5)
    parser.add_argument("--choose-graphs", 
                        help="""IDs of 2 graph (train, test) for E and k (train, valid, test1, test2, ...) for ER 
                                Run print-candidates first for IDs to choose""", 
                        nargs="+", default=[])

    parser.add_argument('--save-as', help="Name of folder", type=str, default=None)
    parser.add_argument("--lcc", help="Only matters in (E) setting. Only apply if orignal graph is 1 LCC", action="store_true", default=False)
    parser.add_argument("--seed", help="Random Seed", type=int, default=42)
    args = parser.parse_args()

    utils.init_seed(args.seed)
    get_new_data(args)


if __name__ == "__main__":
    main()
