"""
Code modified from - https://github.com/bdi-lab/InGram
"""
import os
import sys
import torch
import random
import igraph
import pickle
import argparse
import networkx as nx
import torch_geometric.transforms as T

import utils
from calc_ppr_matrices import *


def create_test(full_data, perc=.90):
    """
    10% is heldout for valid/test
    """
    test = []
    test_graph = []
    test_rel = set()
    test_r2ht = {}
    test_q = {}
    for trip in full_data:
        h, r, t = trip[0], trip[1], trip[2]
        test.append((h,r,t))
        if r in test_r2ht:
            test_r2ht[r].append((h,t))
        else:
            test_r2ht[r] = [(h,t)]
        if (h,'_',t) in test_q:
            test_q[(h,'_',t)].append(r)
        else:
            test_q[(h,'_',t)] = [r]
        test_rel.add(r)
        test_graph.append((h,t))

    G_test = igraph.Graph.TupleList(test_graph, directed = True)
    spanning_test = G_test.spanning_tree()

    num_test = len(test)
    test_msg = set()
    test = set(test)

    for e in spanning_test.es:
        h,t = e.tuple
        h = spanning_test.vs[h]["name"]
        t = spanning_test.vs[t]["name"]
        r = random.choice(test_q[(h,'_',t)])
        test_msg.add((h, r, t))
        test_rel.discard(r)
        test.discard((h,r,t))
    for r in test_rel:
        h,t = random.choice(test_r2ht[r])
        test_msg.add((h,r,t))
        test.discard((h,r,t))
    left_test = sorted(list(test))
    test_msg = sorted(list(test_msg))
    random.shuffle(left_test)
    remainings = int(num_test * perc) - len(test_msg)
    test_msg += left_test[:remainings]
    left_test = left_test[remainings:]

    return test_msg, left_test




def read_KG(dataname):
    """
    Remove reciprocal edges for simplicity

    NOTE: DEAL with GCC here

    Return as list of [(h, r, t), ...]
    """
    dataset = utils.build_dataset(dataname)

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
    not_inv = ~(all_et >= num_rels)
    ei_noninv, et_noninv = all_ei[:, not_inv], all_et[not_inv]

    train_data.edge_index, train_data.edge_type = ei_noninv, et_noninv 
    train_data = T.Compose([T.LargestConnectedComponents()])(train_data)

    trips = []
    for ents, rel in zip(train_data.edge_index.t().tolist(), train_data.edge_type.tolist()):
        trips.append([ents[0], rel, ents[1]])

    return trips



def gather(x):
    ent = []
    rel = []
    for h, r, t in x:
        ent.append(h)
        ent.append(t)
        rel.append(r)
    return list(set(ent)), list(set(rel))

def check_no_overlap(x, y):
    assert len(set(x).intersection(set(y))) == 0
    print("Done: Check no overlap")

def write(path, x):
    with open(path, 'w') as f:
        for h, r, t in x:
            f.write(f"{h}\t{r}\t{t}\n")

def gather_neighbor(triplet, x, thr):
    res = []
    for h, r, t in triplet:
        if h == x:
            res.append(t)
        elif t == x:
            res.append(h)
    res = list(set(res))
    if len(res) > thr:
        res = random.sample(res, thr)
    return res


def sample_2hop(triplet, x, thr):
    sample = set()
    for e in x:
        neighbor = set([e])
        neighbor_1hop = gather_neighbor(triplet, e, thr)  # Returns unique nodes
        neighbor = neighbor.union(set(neighbor_1hop))


        for e1 in neighbor_1hop:
            neighbor_2hop = gather_neighbor(triplet, e1, thr)
            neighbor = neighbor.union(set(neighbor_2hop))

        sample = sample.union(neighbor)

    # print("Final:", len(sample))
    return sample


def merge(x, y, p):
    if p >= 1:
        return y
    elif p <= 0:
        return x
    else:
        num_tot = min(len(x) / (1 - p), len(y) / p)
        random.shuffle(x)
        random.shuffle(y)
        return x[:int(num_tot * (1 - p))] + y[:int(num_tot * p)]

def gcc(triplet):
    edge = []
    for h, r, t in triplet:
        edge.append((h, t))
    G = nx.Graph()
    G.add_edges_from(edge)
    largest_cc = max(nx.connected_components(G), key=len)
    return largest_cc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="FB15k-237")
    parser.add_argument('--ent-train', type=int, default=10)
    parser.add_argument('--ent-test', type=int, default=20)
    parser.add_argument('--tr-sample', type=int, default=50)
    parser.add_argument('--te-sample', type=int, default=50)
    parser.add_argument('--k-hop', type=int, default=2)
    parser.add_argument('--p_rel', type=float, default=1)
    parser.add_argument('--p_tri', type=float, default=0)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--no_save', default=False, action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)

    triplet = read_KG(args.dataset)

    ### Get ents/rels ###
    entity, relation = [], []
    for h, r, t in triplet:
        entity.append(h)
        entity.append(t)
        relation.append(r)

    entity = list(set(entity))
    relation = list(set(relation))

    ### Split relation set into train/valid/test ###
    num_relation = len(relation)
    random.shuffle(relation)
    relation_test = relation_train = relation

    relation_test = set(relation_test)
    relation_train = set(relation_train)

    ### Sample neighbors from train seeds ###
    seed_train = random.sample(entity, args.ent_train)
    entity_train = sample_2hop(triplet, seed_train, args.tr_sample)

    ### Generate train set ###
    train_all = []
    for h, r, t in triplet:
        if h in entity_train and r in relation_train and t in entity_train:
            train_all.append((h, r, t))

    ### Take GCC ###
    gcc_train = gcc(train_all)
    train = []
    for h, r, t in train_all:
        if h in gcc_train:
            train.append((h, r, t))
    random.shuffle(train)

    ### Remove train entities ###
    triplet_p = []
    for h, r, t in triplet:
        if h not in gcc_train and t not in gcc_train:
            triplet_p.append((h, r, t))
    entity_p, relation_p = gather(triplet_p)

    ### Sample neighbors from test seeds ###
    seed_test = random.sample(entity_p, args.ent_test)
    entity_test = sample_2hop(triplet_p, seed_test, args.te_sample)

    ### Generate test set ###
    test_x = []
    test_y = []
    for h, r, t in triplet_p:
        if h in entity_test and r in relation_train and t in entity_test:
            test_x.append((h, r, t))
        elif h in entity_test and r in relation_test and t in entity_test:
            test_y.append((h, r, t))

    ### Merge X_test and Y_test ###
    test_all = merge(test_x, test_y, args.p_tri)

    ### Take GCC ###
    gcc_test = gcc(test_all)
    test = []
    for h, r, t in test_all:
        if h in gcc_test:
            test.append((h, r, t))
    random.shuffle(test)

    ### Check no overlap ###
    check_no_overlap(gcc_train, gcc_test)

    # Remove rels in test that don't show in train
    train_rels = set([t[1] for t in train])
    filtered_test = []
    for t in test:
        if t[1] in train_rels:
            filtered_test.append(t)

    train_graph, final_valid = create_test(train)
    inf_graph, final_test = create_test(filtered_test)

    print("Train/Inf Entities:", len(entity_train), len(entity_test))
    print("Final Train Edges:", len(train_graph))
    # print('Inference Graph b4 Split:', len(inf_graph) + len(final_test))
    print('Inference Graph After Split:', len(inf_graph))
    print('Valid:', len(final_valid))
    print('Test:', len(final_test))

    all_data = {
        "train": train_graph,
        "valid_samples": final_valid,
        "inf_graph": inf_graph,
        "test_samples": final_test
    }
    hyperparams = f"{args.ent_train}-{args.ent_test}-{args.tr_sample}-{args.te_sample}"
    dataset = f"{args.dataset}_generate_{hyperparams}_seed-{args.seed}" 
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "generated", dataset)

    with open(filename + ".pkl", 'wb') as handle:
        pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    alpha, eps = 0.15, 1e-6
    edge_index = torch.Tensor([(int(t[0]), int(t[2])) for t in inf_graph]).t().long()
    num_nodes = torch.max(edge_index).item() + 1

    neighbors, neighbor_weights = get_ppr_matrix(edge_index, num_nodes, alpha, eps)
    sparse_adj = create_sparse_ppr_matrix(neighbors, neighbor_weights)
    save_results(dataset, sparse_adj, alpha, eps, val = False)

   
if __name__ == "__main__":
    main()
