import torch
import random 
import numpy as np 
import networkx as nx 
from tqdm import tqdm
from torch_geometric.data import DataLoader
from scipy.stats import wasserstein_distance

from nbfnet.src.tasks import negative_sampling
from nbfnet.src import datasets as nbf_datasets

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def init_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataset(dataname, version=None, num_test=1, new=False):
    root = "~/datasets/knowledge_graphs/"

    if new:
        root = "~/kg_ppr/new_data/"
        dataset = nbf_datasets.NewSplit(root, dataname)
    elif dataname.lower().startswith("pediatypes") or dataname.lower().startswith("wikitopics"):
        root = "~/kg_ppr/isdea_data/"
        dataset = nbf_datasets.ISDEA_Ind_Datasets(name=dataname, root=root, version=version)
    elif dataname == "FB15k-237" and version is None:
        dataset = nbf_datasets.FB15k237(root)
    elif dataname == "WN18RR" and version is None:
        dataset = nbf_datasets.WN18RR(root)
    elif dataname.lower() == "codex-s":
        dataset = nbf_datasets.CoDExSmall(root=root)
    elif dataname.lower() == "codex-m":
        dataset = nbf_datasets.CoDExMedium(root=root)
    elif dataname.lower() == "codex-l":
        dataset = nbf_datasets.CoDExLarge(root=root)
    elif dataname.lower() == "yago310":
        dataset = nbf_datasets.YAGO310(root=root)
    elif dataname.lower() == "dbpedia100k":
        dataset = nbf_datasets.DBpedia100k(root=root)
    elif dataname.lower() == "hetionet":
        dataset = nbf_datasets.Hetionet(root=root)
    elif dataname in ["FB15k-237", "WN18RR"] and version is not None:
        dataset = nbf_datasets.GrailInductiveDataset(name=dataname, root=root, version=version)
    elif dataname.lower() == "ilpc":
        dataset = nbf_datasets.ILPC2022(root=root, version=version) 
    elif dataname.lower() == "fb-ingram":
        dataset = nbf_datasets.FBIngram(root=root, version=version)
    elif dataname.lower() == "wk-ingram":
        dataset = nbf_datasets.WKIngram(root=root, version=version)
    else:
        raise ValueError("Unknown dataset `%s`" % dataname)

    print("%s dataset" % dataname)
    print("#train: %d, #valid: %d, #test: %d" %
          (dataset[0].target_edge_index.shape[1], dataset[1].target_edge_index.shape[1],
           dataset[2].target_edge_index.shape[1]))

    return dataset


def calc_path_length(data, samples, split="test"):
    """
    It's different for inductive graphs...
    """
    G = nx.Graph()
    G.add_edges_from(data.edge_index.t().tolist())

    all_shortest_lengths = []

    for trip in tqdm(samples, "Calculating Shortest Path"):
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
            except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound):
                l = 100 
        all_shortest_lengths.append(l)
    
    return all_shortest_lengths



def gen_k_negs_per_pos(data, k):
    """
    Generate k negative samples per source and target nodes

    Remove those that are true by iterating through dict
    """
    all_negs = []

    triples = torch.cat([data.target_edge_index, data.target_edge_type.unsqueeze(0)]).t()
    for batch in tqdm(DataLoader(triples, 128), "Generating Negatives"):
        batch = negative_sampling(data, batch, k)
        batch = batch[:, 1:, :] # Remove first which is positive
        batch = batch.reshape(-1, 3)
        batch = batch[:, :2].tolist()
        all_negs.extend(batch)

    # Same for triples with reciprocal relations
    triples = torch.cat([data.target_edge_index[[1, 0]], data.target_edge_type.unsqueeze(0)]).t()
    for batch in tqdm(DataLoader(triples, 128), "Generating Reciprocal Negatives"):
        batch = negative_sampling(data, batch, k)
        batch = batch[:, 1:, :] # Remove first which is positive
        batch = batch.reshape(-1, 3)
        batch = batch[:, :2].tolist()
        all_negs.extend(batch)
    
    return all_negs


def calc_emd_dist(pos_sp, neg_sp, max_sp=10):
    """
    Calc EMD btwn distributions
    """
    # Assign all samples >= max_sp to same bucket
    pos_sp[pos_sp >= max_sp] = max_sp
    neg_sp[neg_sp >= max_sp] = max_sp

    return wasserstein_distance(pos_sp, neg_sp)


def read_data(filename):
    """
    Read triples from file
    """
    trips = []
    with open(filename, "r") as f:
        for l in f:
            u, r, v = l.split()
            trips.append([u, r, v])
    
    return trips